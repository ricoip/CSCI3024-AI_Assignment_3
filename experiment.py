"""Compare PCA variants with 1-nearest-neighbor classification on ORL."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
from itertools import product
from pathlib import Path

import matplotlib
import numpy as np
from scipy.io import loadmat
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits

from rsspca import fit_rsspca, pixel_laplacian

matplotlib.use("Agg")
from matplotlib import pyplot as plt


SEEDS = (11, 23, 37, 53, 71)
COMPONENTS = 30
SPARSITY_GRID = (0.01, 0.1, 0.25)
SMOOTHNESS_GRID = (0.1, 10.0, 100.0)
METHODS = ("PCA", "PCA-L1", "RSPCA", "RSSPCA")


def load_orl(path):
    data = loadmat(path, variable_names=("x", "label"))
    images = data["x"]
    labels = data["label"].ravel()
    _, counts = np.unique(labels, return_counts=True)
    if images.shape != (56, 46, 400) or labels.shape != (400,) or counts.tolist() != [10] * 40:
        raise ValueError("Expected the authors' ORL matrix with 400 images across 40 subjects.")
    samples = np.moveaxis(images, 2, 0).reshape(400, -1)
    return np.ascontiguousarray(samples, dtype=np.float64), labels, images.shape[:2]


def split_by_subject(labels, training_count, seed):
    rng = np.random.default_rng(seed)
    groups = [rng.permutation(np.flatnonzero(labels == person)) for person in np.unique(labels)]
    training = np.sort(np.concatenate([group[:training_count] for group in groups]))
    held_out = np.sort(np.concatenate([group[training_count:] for group in groups]))
    return training, held_out


def prepare_projection(samples, fit_indices, evaluation_indices):
    training = samples[fit_indices]
    mean = training.mean(axis=0)
    scale = training.std(axis=0, ddof=1)
    scale[scale == 0] = 1.0
    train = (training - mean) / scale
    evaluation = (samples[evaluation_indices] - mean) / scale
    _, _, right_vectors = np.linalg.svd(train, full_matrices=False)
    return train, evaluation, right_vectors[:COMPONENTS].T


def accuracy_curve(train, evaluate, basis, train_labels, evaluation_labels):
    train_scores = train @ basis
    evaluate_scores = evaluate @ basis
    accuracies = []
    for count in range(1, basis.shape[1] + 1):
        distances = cdist(evaluate_scores[:, :count], train_scores[:, :count], "sqeuclidean")
        predicted = train_labels[distances.argmin(axis=1)]
        accuracies.append(float(np.mean(predicted == evaluation_labels)))
    return accuracies


def select_penalties(train, validate, pca, laplacian, train_labels, validation_labels, grid):
    candidates = []
    for eta1, eta2 in grid:
        fitted = fit_rsspca(train, pca, laplacian, eta1=eta1, eta2=eta2)
        curve = accuracy_curve(train, validate, fitted.components, train_labels, validation_labels)
        candidates.append({"eta1": eta1, "eta2": eta2, "mean_accuracy": float(np.mean(curve))})
    # Grid order breaks ties in favor of smaller penalties.
    best = max(candidates, key=lambda candidate: candidate["mean_accuracy"])
    return (best["eta1"], best["eta2"]), candidates


def evaluate_split(samples, labels, laplacian, seed):
    outer_train, outer_test = split_by_subject(labels, 7, seed)
    inner_fit, inner_validate = split_by_subject(labels[outer_train], 5, seed + 1000)
    fit_indices = outer_train[inner_fit]
    validation_indices = outer_train[inner_validate]
    train, validate, pca = prepare_projection(samples, fit_indices, validation_indices)
    penalties = {"PCA-L1": (0.0, 0.0)}
    validation = {}
    grids = {"RSPCA": product(SPARSITY_GRID, (0.0,)), "RSSPCA": product(SPARSITY_GRID, SMOOTHNESS_GRID)}
    for method, grid in grids.items():
        penalties[method], validation[method] = select_penalties(
            train, validate, pca, laplacian, labels[fit_indices], labels[validation_indices], grid
        )

    train, test, pca = prepare_projection(samples, outer_train, outer_test)
    results = {"PCA": {"accuracy": accuracy_curve(train, test, pca, labels[outer_train], labels[outer_test])}}
    for method, (eta1, eta2) in penalties.items():
        fitted = fit_rsspca(train, pca, laplacian, eta1=eta1, eta2=eta2)
        results[method] = {
            "eta1": eta1,
            "eta2": eta2,
            "accuracy": accuracy_curve(train, test, fitted.components, labels[outer_train], labels[outer_test]),
            "iterations": fitted.iterations,
            "converged": fitted.converged,
        }
    return {"seed": seed, "validation_seed": seed + 1000, "validation": validation, "methods": results}


def summarize(splits):
    summary = {}
    for method in METHODS:
        curves = np.asarray([split["methods"][method]["accuracy"] for split in splits])
        mean_per_split = curves.mean(axis=1)
        summary[method] = {
            "mean_accuracy": curves.mean(axis=0).tolist(),
            "std_accuracy": curves.std(axis=0, ddof=1).tolist(),
            "mean_over_component_counts": float(mean_per_split.mean()),
            "std_over_splits_of_component_means": float(mean_per_split.std(ddof=1)),
        }
    return summary


def plot_accuracy(summary, destination):
    colors = ("#24364b", "#a45900", "#6a51a3", "#007c78")
    markers = ("o", "s", "^", "D")
    counts = np.arange(1, COMPONENTS + 1)
    figure, axis = plt.subplots(figsize=(8.4, 4.6), layout="constrained")
    for (method, statistics), color, marker in zip(summary.items(), colors, markers, strict=True):
        axis.plot(
            counts, np.array(statistics["mean_accuracy"]) * 100,
            label=method, color=color, marker=marker, markevery=[0, 4, 9, 19, 29],
            markersize=4.5, linewidth=1.8,
        )
    axis.set(
        title="ORL face recognition accuracy",
        xlabel="Number of projection components",
        ylabel="Mean test accuracy (%)",
        xticks=[1, 5, 10, 15, 20, 25, 30],
        ylim=(0, 100), xlim=(0.5, 30.5),
    )
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="lower right", frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/ORL.mat"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    if not args.data.is_file():
        parser.error(f"Dataset missing at {args.data}. See README.md for the download command.")

    samples, labels, image_shape = load_orl(args.data)
    laplacian = pixel_laplacian(*image_shape)
    splits = []
    with threadpool_limits(limits=1, user_api="blas"):
        for seed in SEEDS:
            print(f"Evaluating outer split {seed}", flush=True)
            splits.append(evaluate_split(samples, labels, laplacian, seed))

    summary = summarize(splits)
    evidence = {
        "dataset_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "test_images_per_split": 120,
        "components": list(range(1, COMPONENTS + 1)),
        "runtime": {
            "python": platform.python_version(),
            "system": platform.system(),
            "machine": platform.machine(),
            "packages": {
                package: importlib.metadata.version(package)
                for package in ("numpy", "scipy", "matplotlib", "threadpoolctl")
            },
        },
        "splits": splits,
        "summary": summary,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    plot_accuracy(summary, args.output / "accuracy.png")
    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
