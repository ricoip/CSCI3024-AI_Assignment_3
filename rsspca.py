"""RSSPCA projection directions from Wang et al. (2025), Algorithm 1."""

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import factorized, spsolve


@dataclass(frozen=True)
class Projection:
    components: np.ndarray
    iterations: tuple[int, ...]
    converged: tuple[bool, ...]


def pixel_laplacian(height: int, width: int) -> sparse.csc_matrix:
    """Build D - A for four-neighbor connectivity in row-major pixel order."""
    if height < 1 or width < 1:
        raise ValueError("Image dimensions must be positive.")
    pixels = np.arange(height * width).reshape(height, width)
    start = np.concatenate((pixels[:, :-1].ravel(), pixels[:-1, :].ravel()))
    end = np.concatenate((pixels[:, 1:].ravel(), pixels[1:, :].ravel()))
    adjacency = sparse.csc_matrix(
        (np.ones(2 * start.size), (np.r_[start, end], np.r_[end, start])),
        shape=(pixels.size, pixels.size),
    )
    degrees = np.asarray(adjacency.sum(axis=1)).ravel()
    return sparse.diags(degrees, format="csc") - adjacency


def fit_rsspca(
    samples: np.ndarray,
    initial_components: np.ndarray,
    laplacian: sparse.csc_matrix,
    *,
    eta1: float,
    eta2: float,
    tolerance: float = 1e-4,
    max_iterations: int = 100,
) -> Projection:
    """Fit directions from centered sample rows and ordinary PCA directions.

    The columns of initial_components set the component count and initialize
    the corresponding iterations. Setting both penalties to zero gives PCA-L1.
    """
    residual = np.asarray(samples, dtype=np.float64)
    if residual.ndim != 2 or not np.isfinite(residual).all():
        raise ValueError("Samples must be a finite two-dimensional array.")
    feature_count = residual.shape[1]
    initial_components = np.asarray(initial_components, dtype=np.float64)
    if (
        initial_components.ndim != 2
        or initial_components.shape[0] != feature_count
        or not 0 < initial_components.shape[1] <= min(residual.shape)
        or not np.isfinite(initial_components).all()
    ):
        raise ValueError("Supply finite PCA directions as columns, one per component.")
    if laplacian.shape != (feature_count, feature_count):
        raise ValueError("The Laplacian must match the number of features.")
    if not np.isfinite([eta1, eta2, tolerance]).all() or min(eta1, eta2) < 0:
        raise ValueError("Penalties must be finite and nonnegative.")
    if tolerance <= 0 or max_iterations < 1:
        raise ValueError("The tolerance and iteration limit must be positive.")

    if eta1 == 0 and eta2 > 0:
        solve_smooth = factorized(sparse.eye(feature_count, format="csc") + eta2 * laplacian)
    elif eta1 > 0 and eta2 > 0:
        smooth_penalty = (eta2 * laplacian).tocoo()

    components = np.empty_like(initial_components)
    iterations = []
    converged = []
    for index, direction in enumerate(initial_components.T):
        scores = residual @ direction
        objective = np.abs(scores).sum()
        if objective == 0:
            raise ValueError("A PCA initializer has zero projection on the residual data.")

        for iteration in range(1, max_iterations + 1):
            gradient = residual.T @ np.sign(scores)
            if eta1 > 0 and eta2 > 0:
                # With S = sqrt(U), Eq. (32) is equivalent to
                # S (U + eta1 I + eta2 S L S)^(-1) S v. This works even at zeros.
                magnitude = np.abs(direction)
                root = np.sqrt(magnitude)
                weights = smooth_penalty.data * root[smooth_penalty.row] * root[smooth_penalty.col]
                system = sparse.csc_matrix(
                    (weights, (smooth_penalty.row, smooth_penalty.col)), shape=laplacian.shape
                )
                system.setdiag(system.diagonal() + magnitude + eta1)
                update = root * spsolve(system, root * gradient)
            elif eta1 > 0:
                magnitude = np.abs(direction)
                update = magnitude / (magnitude + eta1) * gradient
            elif eta2 > 0:
                update = solve_smooth(gradient)
            else:
                update = gradient

            length = np.linalg.norm(update)
            if not np.isfinite(length) or length == 0:
                raise ValueError("The RSSPCA update did not yield a finite nonzero direction.")
            direction = update / length
            scores = residual @ direction
            new_objective = np.abs(scores).sum()
            relative_change = abs(new_objective - objective) / objective
            if relative_change <= tolerance:
                break
            objective = new_objective

        components[:, index] = direction
        iterations.append(iteration)
        converged.append(bool(relative_change <= tolerance))
        if index + 1 < initial_components.shape[1]:
            # (I - W W^T) X from Algorithm 1, expressed with sample rows.
            basis = components[:, :index + 1]
            residual = residual - (residual @ basis) @ basis.T

    return Projection(components, tuple(iterations), tuple(converged))
