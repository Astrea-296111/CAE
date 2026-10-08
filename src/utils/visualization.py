from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cae-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/cae-cache")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator
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
    edges = edges[edges[:, 0] < edges[:, 1]]
    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.add_collection3d(Line3DCollection(pos[edges], colors="#a3a8aa", linewidths=0.5, alpha=0.45))
    if values is None:
        for mask, color, label in [(~graph.boundary.numpy(), "#4772b2", "Free node"),
                                    (graph.boundary.numpy(), "#bd4343", "Fixed boundary")]:
            ax.scatter(*pos[mask].T, c=color, s=20, depthshade=False, label=label)
        ax.legend(loc="upper left", bbox_to_anchor=(0.94, 0.9), frameon=False)
    else:
        points = ax.scatter(*pos.T, c=np.asarray(values).ravel(), cmap="coolwarm", s=20, depthshade=False)
        fig.colorbar(points, ax=ax, shrink=0.65, label="u_z (m)", pad=0.15)
    ax.set(xlabel="x (m)", ylabel="y (m)", zlabel="z (m)", title=title)
    ax.set_box_aspect(np.maximum(np.ptp(pos, axis=0), 0.1))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_major_locator(MaxNLocator(4))
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
            points = ax.scatter(*pos.T, c=values, cmap="coolwarm", norm=norm, s=20, depthshade=False)
            ax.set(xlabel="x (m)", ylabel="y (m)", zlabel="z (m)", title=f"Synthetic: {title}")
            ax.set_box_aspect(np.maximum(np.ptp(pos, axis=0), 0.1))
            for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                axis.set_major_locator(MaxNLocator(3))
        fig.subplots_adjust(left=0.02, right=0.85, wspace=0.2)
        color_axis = fig.add_axes([0.93, 0.22, 0.015, 0.56])
        fig.colorbar(points, cax=color_axis, label="u_z (m)")
        save_figure(fig, output_dir / "displacement_comparison.png")
