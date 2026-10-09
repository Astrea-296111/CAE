from copy import deepcopy
import json

import numpy as np
import pytest

from src.data.candidates import candidate_mesh, generate_candidates, validate_manifest
from src.data.contract import feature_schema, fit_training_domain, schema_hash, validate_checkpoint_contract
from src.data.graph_builder import build_graph
from src.data.synthetic_solver import SyntheticPhysicsSolver
from src.inference.pipeline import run_screening
from src.inference.reliability import ReliabilityPolicy, assess_domain, assess_prediction


@pytest.fixture
def screening_case(config, monkeypatch):
    manifest = generate_candidates(config, count=8, seed=22, ood_fraction=0)
    meshes = [candidate_mesh(row) for row in manifest["candidates"]]
    graphs = [build_graph(mesh) for mesh in meshes]
    checkpoint = {"checkpoint_version": 1, "feature_schema": feature_schema(),
                  "feature_schema_sha256": schema_hash(), "training_domain": fit_training_domain(graphs),
                  "kind": "physics", "config": config, "dataset_sha256": "fixture"}

    def predictor(model, batch, batch_size):
        results = [SyntheticPhysicsSolver().run(meshes[int(g.sample_id.split("-")[-1]) - 1]) for g in batch]
        return np.array([r.stiffness for r in results]), [r.displacement for r in results]

    monkeypatch.setattr("src.inference.pipeline.predict_graphs", predictor)
    return manifest, meshes, checkpoint


class CountingSolver:
    def __init__(self, fail=False):
        self.calls = 0
        self.fail = fail

    def run(self, mesh):
        self.calls += 1
        if self.fail:
            raise RuntimeError("solver unavailable")
        return SyntheticPhysicsSolver().run(mesh)


def test_label_free_candidates_reproduce_topology(config):
    first = generate_candidates(config, count=10, seed=17)
    assert first == generate_candidates(config, count=10, seed=17)
    row = first["candidates"][0]
    a, b = candidate_mesh(row), candidate_mesh(row)
    assert np.array_equal(a.pos, b.pos) and np.array_equal(a.edges, b.edges)
    assert "y_k" not in build_graph(a) and "y_u" not in build_graph(a)
    row["y_k"] = 1
    with pytest.raises(ValueError, match="Candidate fields"):
        candidate_mesh(row)


def test_duplicate_ids_and_bad_geometry_are_rejected(config):
    manifest = generate_candidates(config, count=2)
    manifest["candidates"][1]["sample_id"] = manifest["candidates"][0]["sample_id"]
    with pytest.raises(ValueError, match="Duplicate"):
        validate_manifest(manifest)
    row = manifest["candidates"][0]
    row["grid_shape"] = [10000000, 3, 4]
    with pytest.raises(ValueError, match="grid_shape"):
        candidate_mesh(row)


def test_top_k_uses_only_selected_solver_calls(screening_case, tmp_path):
    manifest, _, checkpoint = screening_case
    solver = CountingSolver()
    result = run_screening(None, checkpoint, manifest, tmp_path, top_k=2, solver=solver)
    records = json.loads((tmp_path / "predictions.json").read_text())
    assert solver.calls == result["screening_solver_calls"] == 2
    assert result["solver_call_reduction_percent"] == 75
    assert result["audit_metrics"] is None and not (tmp_path / "oracle.csv").exists()
    assert sum(r["status"] == "screened_out" for r in records) == 6
    assert sum(r["final_rank"] is not None for r in records) == 2
    assert len(np.load(tmp_path / "displacements.npz").files) == 8


def test_audit_is_separate_and_does_not_change_routing(screening_case, tmp_path):
    manifest, _, checkpoint = screening_case
    normal = run_screening(None, checkpoint, manifest, tmp_path / "normal", top_k=2)
    audit = run_screening(None, checkpoint, manifest, tmp_path / "audit", top_k=2, audit=True)
    assert normal["screening_solver_calls"] == audit["screening_solver_calls"] == 2
    assert audit["additional_oracle_audit_calls"] == 6
    assert audit["total_solver_calls"] == 8
    assert audit["audit_metrics"]["verified_top_k_recall"] == 1
    assert audit["verified_count"] == 2
    assert (tmp_path / "normal/predictions.json").read_text() == (tmp_path / "audit/predictions.json").read_text()


def test_risk_fallback_and_invalid_input_are_distinct(screening_case, tmp_path):
    manifest, meshes, checkpoint = screening_case
    # Rebuild the fixture's first mesh so mocked predictions match the OOD input.
    manifest["candidates"][0]["parameters"]["thickness"] = 1.35
    meshes[0] = candidate_mesh(manifest["candidates"][0])
    manifest["candidates"][1]["parameters"]["thickness"] = -1
    result = run_screening(None, checkpoint, manifest, tmp_path, top_k=2, audit=True)
    records = json.loads((tmp_path / "predictions.json").read_text())
    assert result["fallback_count"] == 1 and result["invalid_candidate_count"] == 1
    assert result["screening_solver_calls"] == 3
    assert records[0]["route"] == "fallback" and "ood:thickness" in records[0]["reasons"]
    assert records[1]["status"] == "invalid_input" and records[1]["verified_stiffness"] is None
    assert result["audit_metrics"] is None  # No full-pool recall on an incomplete pool.


def test_failed_solver_is_not_a_verified_result(screening_case, tmp_path):
    manifest, _, checkpoint = screening_case
    result = run_screening(None, checkpoint, manifest, tmp_path, top_k=2, solver=CountingSolver(fail=True), audit=True)
    records = json.loads((tmp_path / "predictions.json").read_text())
    assert result["screening_solver_failures"] == 2 and result["verified_count"] == 0
    assert all(r["final_rank"] is None for r in records)
    assert result["audit_metrics"] is None
    assert result["audit_solver_failures"] == 6


def test_model_failure_falls_back_for_every_valid_input(screening_case, tmp_path, monkeypatch):
    manifest, _, checkpoint = screening_case
    def fail(*args):
        raise RuntimeError("inference unavailable")
    monkeypatch.setattr("src.inference.pipeline.predict_graphs", fail)
    result = run_screening(None, checkpoint, manifest, tmp_path, top_k=2)
    assert result["fallback_count"] == result["screening_solver_calls"] == 8
    assert result["verified_count"] == 8 and result["selected_top_k"] == 0
    assert result["solver_call_reduction_percent"] == 0


def test_nonfinite_prediction_is_serializable_and_routes_to_solver(screening_case, tmp_path, monkeypatch):
    manifest, _, checkpoint = screening_case
    monkeypatch.setattr("src.inference.pipeline.predict_graphs",
                        lambda model, graphs, size: (np.full(len(graphs), np.nan), None))
    result = run_screening(None, checkpoint, manifest, tmp_path, top_k=2)
    assert result["fallback_count"] == 8
    text = (tmp_path / "predictions.json").read_text()
    assert "NaN" not in text and "Infinity" not in text


def test_schema_mismatch_and_legacy_fail_before_solver(screening_case, tmp_path):
    manifest, _, checkpoint = screening_case
    wrong = deepcopy(checkpoint)
    wrong["feature_schema"]["node_features"].reverse()
    solver = CountingSolver()
    with pytest.raises(ValueError, match="schema"):
        run_screening(None, wrong, manifest, tmp_path, solver=solver, top_k=2)
    assert solver.calls == 0
    validate_checkpoint_contract({})  # Evaluation retains explicit v0 compatibility.
    with pytest.raises(ValueError, match="retrain"):
        validate_checkpoint_contract({}, require_contract=True)


def test_physics_sign_boundary_and_missing_field_gate(screening_case):
    _, meshes, checkpoint = screening_case
    mesh = meshes[0]
    solution = SyntheticPhysicsSolver().run(mesh)
    domain, policy = checkpoint["training_domain"], ReliabilityPolicy()
    assert assess_prediction(mesh, solution.stiffness, solution.displacement, domain, policy)["reasons"] == []
    bad = assess_prediction(mesh, solution.stiffness, -solution.displacement, domain, policy)
    assert "nonpositive_angle" in bad["reasons"] and "physics_residual" in bad["reasons"]
    moved = solution.displacement.copy()
    moved[mesh.boundary] = 0.001
    assert "boundary_displacement" in assess_prediction(mesh, solution.stiffness, moved, domain, policy)["reasons"]
    assert "missing_displacement" in assess_prediction(mesh, solution.stiffness, None, domain, policy)["reasons"]


def test_output_directory_cannot_reuse_old_results(screening_case, tmp_path):
    manifest, _, checkpoint = screening_case
    (tmp_path / "old.json").write_text("{}")
    with pytest.raises(ValueError, match="not empty"):
        run_screening(None, checkpoint, manifest, tmp_path, top_k=2)


def test_constant_training_parameters_allow_float32_rounding(screening_case):
    _, meshes, _ = screening_case
    mesh = meshes[0]
    domain = fit_training_domain([build_graph(mesh)])
    assert assess_domain(mesh, domain, ReliabilityPolicy(domain_margin=0)) == []
    mesh.design_parameters["thickness"] *= 1.01
    assert "ood:thickness" in assess_domain(mesh, domain, ReliabilityPolicy(domain_margin=0))
