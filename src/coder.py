""""""

import sys
from pathlib import Path

sys.path.append(str(Path(sys.path[0]).parent))

from sklearn.linear_model import Lasso, OrthogonalMatchingPursuit
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from numpy.linalg import norm
from typing import Any
import numpy as np

from src.utils import convert_output


class SparseCoder:
    def __init__(
        self,
        X: np.ndarray,
        D: np.ndarray = None,
        params: dict[Any] = {},
    ):
        # ================================================================
        #                          Parameters
        # ================================================================
        defaults = {
            'explained_variance': None,
            'sparsity_level': None,
            'regularizer': 1e-3,
            'max_iter': 1,
            'tol': 1e-3,
        }
        defaults.update(params)
        self.tol = defaults['tol']
        self.alpha = defaults['regularizer']
        self.max_iter = defaults['max_iter']
        self.ev = defaults['explained_variance']

        # ================================================================
        #                          Variables
        # ================================================================
        self.X = X
        self.stalk_dim, self.n_examples = self.X.shape
        self.D = D
        self.n_atoms = self.D.shape[1] if self.D is not None else None
        try:
            self.k0 = int(
                np.floor(defaults['sparsity_level'] * self.stalk_dim)
            )
        except TypeError:
            self.k0 = defaults['sparsity_level']

    def _set_dictionary(
        self,
        normalize: bool = False,
    ) -> None:
        if normalize:
            sc = StandardScaler()
            self.X = sc.fit_transform(self.X)
        pca = PCA(
            n_components=self.ev,
            svd_solver='full',
        )
        pca.fit(self.X.T)
        self.D = pca.components_.T
        self.S = pca.transform(self.X.T).T
        return None

    def _sparse_coding_lasso(self) -> None:
        """
        Compute sparse codes via Lasso for multiple signals.
        Solves, for each column x_j of Y:
            minimize_x  0.5 * ||y_j - D @ x||_2^2  +  alpha * ||x||_1
        """

        lasso = Lasso(
            alpha=self.alpha,
            fit_intercept=False,
            max_iter=self.max_iter,
            tol=self.tol,
        )
        lasso.fit(self.D.real, self.X)
        self.S = lasso.coef_.T

        return None

    def _sparse_coding_omp(self) -> None:
        """
        Compute sparse codes via Orthogonal Matching Pursuit (OMP) for multiple signals.
        Solves, for each column x_j of X:
            minimize_s ||x_j - D @ s||_2  subject to ||x_j - D @ s||_2 <= tol
        or stops after selecting up to max_iter atoms.
        """
        dd = norm(self.D, axis=0)
        Domp = self.D @ np.diag(1.0 / dd)
        omp = OrthogonalMatchingPursuit(
            n_nonzero_coefs=self.k0,
            tol=self.tol,
            fit_intercept=False,
            precompute=True,
        )
        omp.fit(Domp.real, self.X)
        self.S = omp.coef_.T

        return None

    @convert_output
    def fit(self) -> tuple[np.ndarray, np.ndarray]:
        if self.D is None:
            self._set_dictionary()
        else:
            if self.k0 is None:
                self._sparse_coding_lasso()
            else:
                self._sparse_coding_omp()
        return self.D, self.S
