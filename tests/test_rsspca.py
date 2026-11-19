import unittest

import numpy as np

from rsspca import fit_rsspca, pixel_laplacian


class ProjectionTests(unittest.TestCase):
    def test_laplacian_preserves_pixel_geometry(self):
        laplacian = pixel_laplacian(2, 3).toarray()
        np.testing.assert_array_equal(laplacian.sum(axis=1), np.zeros(6))
        np.testing.assert_array_equal(laplacian, laplacian.T)
        np.testing.assert_array_equal(laplacian[2], [0, -1, 2, 0, 0, -1])
        np.testing.assert_array_equal(laplacian[4], [0, -1, 0, -1, 3, -1])

    def test_sparse_update_matches_equation_32_at_zero_weight(self):
        samples = np.array([[3, 1, 2, 0], [-1, 3, -1, 4], [-2, -4, -1, -4]])
        initial = np.array([0.8, 0.0, -0.6, 0.0])
        laplacian = pixel_laplacian(2, 2)
        eta1, eta2 = 0.17, 2.0
        diagonal = np.diag(np.abs(initial))
        gradient = samples.T @ np.sign(samples @ initial)
        system = diagonal + eta1 * np.eye(4) + eta2 * diagonal @ laplacian.toarray()
        expected = np.linalg.solve(system, diagonal @ gradient)
        actual = fit_rsspca(
            samples, initial[:, None], laplacian, eta1=eta1, eta2=eta2, max_iterations=1
        )
        np.testing.assert_allclose(actual.components[:, 0], expected / np.linalg.norm(expected), atol=1e-14)
        self.assertEqual(actual.iterations, (1,))
        self.assertEqual(actual.converged, (False,))

    def test_zero_penalties_give_normalized_l1_update(self):
        samples = np.array([[3, 1, -1], [-1, 3, 2], [-2, -4, -1]])
        initial = np.array([1.0, 0.0, 0.0])
        expected = samples.T @ np.sign(samples @ initial)
        actual = fit_rsspca(
            samples, initial[:, None], pixel_laplacian(1, 3), eta1=0.0, eta2=0.0, max_iterations=1
        )
        np.testing.assert_allclose(actual.components[:, 0], expected / np.linalg.norm(expected))


if __name__ == "__main__":
    unittest.main()
