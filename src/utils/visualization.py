from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Line3DCollection


plt.rcParams.update({"font.size": 10, "figure.dpi": 130, "axes.spines.top": False,
                     "axes.spines.right": False})


def save_figure(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_mesh(graph, path, values=None, title="Synthetic elastic network"):
    pos = graph.pos.numpy()
    edges = graph.edge_index.numpy().T
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.add_collection3d(Line3DCollection(pos[edges], colors="#a3a8aa", linewidths=0.5, alpha=0.45))
    color = values if values is not None else graph.boundary.numpy().astype(float)
    points = ax.scatter(*pos.T, c=np.asarray(color).ravel(), cmap="coolwarm", s=14)
    fig.colorbar(points, ax=ax, shrink=0.65, label="u_z (m)" if values is not None else "fixed boundary")
    ax.set(xlabel="x (m)", ylabel="y (m)", zlabel="z (m)", title=title)
    ax.set_box_aspect(np.maximum(np.ptp(pos, axis=0), 0.1))
    save_figure(fig, path)


def plot_dataset(dataset, output_dir):
    output_dir = Path(output_dir)
    plot_mesh(dataset["train"][0], output_dir / "synthetic_mesh.png")
    fig, ax = plt.subplots(figsize=(7, 4))
    for split, color in [("train", "#16877e"), ("test", "#657a86"), ("ood", "#c65d48")]:
        values = [g.y_k.item() for g in dataset[split]]
        ax.hist(values, bins=20, alpha=0.55, label=split, color=color)
    ax.set(xlabel="Torsional stiffness (N m/rad)", ylabel="Design count",
           title="Synthetic label distribution")
    ax.legend()
    save_figure(fig, output_dir / "label_distribution.png")


def plot_evaluation(graphs, truth, prediction, displacement, output_dir):
    output_dir = Path(output_dir)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(truth, prediction, c="#16877e", s=22)
    low, high = min(truth.min(), prediction.min()), max(truth.max(), prediction.max())
    ax.plot([low, high], [low, high], color="#a34848", linestyle="--")
    ax.set(xlabel="True stiffness (N m/rad)", ylabel="Predicted stiffness (N m/rad)",
           title="Synthetic experiment: prediction vs truth")
    save_figure(fig, output_dir / "prediction_vs_truth.png")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(100 * (prediction - truth) / truth, bins=15, color="#657a86")
    ax.set(xlabel="Relative error (%)", ylabel="Design count", title="Synthetic prediction error")
    save_figure(fig, output_dir / "error_distribution.png")
    fig, ax = plt.subplots(figsize=(5, 5))
    true_rank = np.argsort(np.argsort(-truth, kind="stable"), kind="stable") + 1
    pred_rank = np.argsort(np.argsort(-prediction, kind="stable"), kind="stable") + 1
    ax.scatter(true_rank, pred_rank, color="#16877e", s=20)
    ax.plot([1, len(truth)], [1, len(truth)], "--", color="#a34848")
    ax.set(xlabel="True rank (1 = best)", ylabel="Predicted rank", title="Synthetic design ranking")
    save_figure(fig, output_dir / "ranking.png")
    if displacement is not None:
        graph = graphs[0]
        pos = graph.pos.numpy()
        true_u, pred_u = graph.y_u.numpy().ravel(), displacement[0].ravel()
        limit = max(np.abs(true_u).max(), np.abs(pred_u).max(), 1e-12)
        norm = Normalize(-limit, limit)
        fig = plt.figure(figsize=(12, 4))
        for number, values, title in [(1, true_u, "True u_z"), (2, pred_u, "Predicted u_z")]:
            ax = fig.add_subplot(1, 2, number, projection="3d")
            ax.add_collection3d(Line3DCollection(
                pos[graph.edge_index.numpy().T], colors="#a3a8aa", linewidths=0.4, alpha=0.3
            ))
            points = ax.scatter(*pos.T, c=values, cmap="coolwarm", norm=norm, s=15)
            ax.set(xlabel="x (m)", ylabel="y (m)", zlabel="z (m)", title=f"Synthetic: {title}")
            ax.set_box_aspect(np.maximum(np.ptp(pos, axis=0), 0.1))
        fig.colorbar(points, ax=fig.axes, shrink=0.7, label="u_z (m)", pad=0.08)
        save_figure(fig, output_dir / "displacement_comparison.png")
