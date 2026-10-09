"""Predict once, verify Top-K and risk cases, optionally audit the full pool."""

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from .reliability import ReliabilityPolicy, assess_domain, assess_prediction
from ..cae.synthetic_adapter import SyntheticCAESolver
from ..data.candidates import candidate_mesh, validate_manifest
from ..data.contract import validate_checkpoint_contract
from ..data.graph_builder import build_graph
from ..training.evaluator import load_checkpoint, predict_graphs
from ..training.metrics import ranking_metrics, regression_metrics
from ..utils.logging import write_json


def run_screening(model, checkpoint, manifest, output_dir, top_k=10, policy=None,
                  solver=None, audit=False, checkpoint_sha256=None):
    """Solver failures stay explicit; only successful solver outputs get final ranks."""
    started = perf_counter()
    validate_checkpoint_contract(checkpoint, require_contract=True)
    validate_manifest(manifest)
    rows = manifest["candidates"]
    if type(top_k) is not int or not 1 <= top_k <= len(rows):
        raise ValueError("top_k must be between 1 and the candidate count")
    batch_size = int(checkpoint["config"]["training"]["batch_size"])
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    policy = policy or ReliabilityPolicy()
    solver = solver or SyntheticCAESolver()
    output_dir = Path(output_dir)
    # Require a fresh run directory so old displacement files cannot masquerade as new output.
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Output directory is not empty; choose a new run directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    input_hash = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    records, meshes, graphs, valid_indices, events, fields = [], {}, [], [], [], {}
    for i, row in enumerate(rows):
        record = {"sample_id": row["sample_id"], "status": "pending", "route": None,
                  "reasons": [], "predicted_stiffness": None, "angle_rad": None,
                  "physics_residual": None, "surrogate_rank": None, "final_rank": None,
                  "verified_stiffness": None, "solver_error": None,
                  "inference_error": None, "validation_error": None,
                  "displacement_key": None}
        records.append(record)
        try:
            mesh = candidate_mesh(row, policy.max_nodes)
            graph = build_graph(mesh, sample_id=row["sample_id"])
            if not np.isfinite(graph.x.numpy()).all() or not np.isfinite(graph.edge_attr.numpy()).all():
                raise ValueError("Nonfinite graph features")
            if not np.isfinite([mesh.span, mesh.torque]).all() or min(mesh.span, mesh.torque) <= 0:
                raise ValueError("Span and torque must be finite and positive")
            meshes[i] = mesh
            record["reasons"] = assess_domain(mesh, checkpoint["training_domain"], policy)
            graphs.append(graph)
            valid_indices.append(i)
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            record.update(status="invalid_input", route="reject", reasons=["invalid_input"],
                          validation_error=str(exc))
    graph_seconds = perf_counter() - started
    inference_start = perf_counter()
    for offset in range(0, len(graphs), batch_size):
        batch_graphs = graphs[offset:offset + batch_size]
        indices = valid_indices[offset:offset + batch_size]
        try:
            predictions, displacement = predict_graphs(model, batch_graphs, batch_size)
            predictions = np.asarray(predictions).reshape(-1)
            if len(predictions) != len(indices) or (displacement is not None and len(displacement) != len(indices)):
                raise ValueError("Model output cardinality does not match the batch")
        except Exception as exc:
            # Acceleration can fail without preventing solver verification of valid inputs.
            for i in indices:
                records[i]["reasons"].append("inference_failed")
                records[i]["inference_error"] = str(exc)
            continue
        for j, i in enumerate(indices):
            k = float(predictions[j])
            u = None if displacement is None else displacement[j]
            assessment = assess_prediction(meshes[i], k, u, checkpoint["training_domain"], policy)
            records[i].update(assessment)
            records[i]["predicted_stiffness"] = k if np.isfinite(k) else None
            if u is not None and np.shape(u) == (len(meshes[i].pos), 1) and np.isfinite(u).all():
                key = f"candidate_{i:06d}"
                fields[key] = u
                records[i]["displacement_key"] = key
    inference_seconds = perf_counter() - inference_start
    reliable = [i for i in valid_indices if not records[i]["reasons"]]
    reliable.sort(key=lambda i: -records[i]["predicted_stiffness"])
    selected = set(reliable[:top_k])
    for rank, i in enumerate(reliable, 1):
        records[i]["surrogate_rank"] = rank
    for i in valid_indices:
        record = records[i]
        if record["reasons"]:
            record.update(route="fallback", status="awaiting_solver")
        elif i in selected:
            record.update(route="top_k", status="awaiting_solver")
        else:
            record.update(route="defer", status="screened_out")

    events.extend({"sample_id": r["sample_id"], "stage": "routing", "route": r["route"],
                   "reasons": r["reasons"], "status": r["status"]} for r in records)

    calls, failures = 0, 0
    truth = {}
    solver_start = perf_counter()
    for i in valid_indices:
        record = records[i]
        if record["route"] == "defer":
            continue
        calls += 1
        try:
            result = solver.run(meshes[i])
            if not np.isfinite(result.stiffness) or result.stiffness <= 0:
                raise ValueError("Solver returned invalid stiffness")
            truth[i] = float(result.stiffness)
            record.update(status="verified", verified_stiffness=truth[i])
        except Exception as exc:
            failures += 1
            record.update(status="solver_failed", solver_error=str(exc))
        events.append({"sample_id": record["sample_id"], "stage": "verification",
                       "status": record["status"], "error": record["solver_error"]})
    solver_seconds = perf_counter() - solver_start
    verified_order = sorted(truth, key=lambda i: -truth[i])
    for rank, i in enumerate(verified_order, 1):
        records[i]["final_rank"] = rank

    audit_calls, audit_failures, audit_metrics = 0, 0, None
    audit_start = perf_counter()
    if audit:
        for i in valid_indices:
            if i in truth or records[i]["route"] != "defer":
                continue  # A failed verification is not silently retried by the audit.
            audit_calls += 1
            try:
                result = solver.run(meshes[i])
                if not np.isfinite(result.stiffness) or result.stiffness <= 0:
                    raise ValueError("Solver returned invalid stiffness")
                truth[i] = float(result.stiffness)
            except Exception as exc:
                audit_failures += 1
                events.append({"sample_id": records[i]["sample_id"], "stage": "audit",
                               "status": "solver_failed", "error": str(exc)})
        complete = len(truth) == len(rows) and all(r["predicted_stiffness"] is not None for r in records)
        if complete:
            targets = np.array([truth[i] for i in range(len(rows))])
            predictions = np.array([r["predicted_stiffness"] for r in records])
            true_top = set(np.argsort(-targets, kind="stable")[:top_k].tolist())
            audit_metrics = {
                **regression_metrics(targets, predictions), **ranking_metrics(targets, predictions),
                "verified_top_k_recall": len(true_top & set(verified_order)) / top_k,
            }
        pd.DataFrame([{"sample_id": rows[i]["sample_id"], "oracle_stiffness": value}
                      for i, value in sorted(truth.items())]).to_csv(output_dir / "oracle.csv", index=False)
    audit_seconds = perf_counter() - audit_start
    reasons = Counter(reason for record in records for reason in record["reasons"])
    valid_count = len(valid_indices)
    fallback_count = sum(r["route"] == "fallback" for r in records)
    residuals = [r["physics_residual"] for r in records if r["physics_residual"] is not None]
    inference_failures = sum("inference_failed" in r["reasons"] for r in records)
    invalid_predictions = sum(any(reason in {"invalid_stiffness", "invalid_displacement", "inference_failed"}
                                 for reason in r["reasons"]) for r in records)
    ood_count = sum(any(reason.startswith("ood:") for reason in r["reasons"]) for r in records)
    summary = {
        "provenance": "synthetic-only", "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint_sha256": checkpoint_sha256, "dataset_sha256": checkpoint.get("dataset_sha256"),
        "feature_schema_sha256": checkpoint["feature_schema_sha256"],
        "candidate_manifest_sha256": input_hash, "model_kind": checkpoint["kind"],
        "policy": policy.to_dict(), "candidate_count": len(rows), "valid_candidate_count": valid_count,
        "invalid_candidate_count": len(rows) - valid_count, "requested_top_k": top_k,
        "selected_top_k": len(selected), "fallback_count": fallback_count,
        "fallback_rate": fallback_count / valid_count if valid_count else None,
        "ood_count": ood_count, "ood_rate": ood_count / valid_count if valid_count else None,
        "inference_failed_count": inference_failures,
        "inference_success_rate": 1 - invalid_predictions / valid_count if valid_count else None,
        "reason_counts": dict(reasons), "screening_solver_calls": calls,
        "screening_solver_failures": failures, "verified_count": len(verified_order),
        "solver_call_reduction_percent": 100 * (1 - calls / valid_count) if valid_count else None,
        "mean_physics_residual": float(np.mean(residuals)) if residuals else None,
        "audit_enabled": audit, "additional_oracle_audit_calls": audit_calls,
        "audit_solver_failures": audit_failures, "total_solver_calls": calls + audit_calls,
        "audit_metrics": audit_metrics,
        "timing_seconds": {"graph_build": graph_seconds, "inference": inference_seconds,
                           "verification": solver_seconds, "audit": audit_seconds,
                           "compute_total": perf_counter() - started},
    }
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "predictions.json", records)
    write_json(output_dir / "verification_queue.json", {
        "schema_version": 1, "provenance": "synthetic-only",
        "candidates": [row for row, record in zip(rows, records) if record["route"] in {"top_k", "fallback"}],
    })
    table = pd.DataFrame([{**r, "reasons": ";".join(r["reasons"])} for r in records])
    table.to_csv(output_dir / "candidates.csv", index=False)
    table[table["status"] == "verified"].sort_values("final_rank").to_csv(
        output_dir / "verified.csv", index=False)
    if fields:
        np.savez_compressed(output_dir / "displacements.npz", **fields)
    (output_dir / "events.jsonl").write_text(
        "".join(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n" for event in events), encoding="utf-8")
    return summary


def screen_checkpoint(checkpoint_path, manifest, output_dir, **kwargs):
    path = Path(checkpoint_path)
    model, checkpoint = load_checkpoint(path, require_contract=True)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return run_screening(model, checkpoint, manifest, output_dir,
                         checkpoint_sha256=digest, **kwargs)
