""""""

import sys
from pathlib import Path

sys.path.append(str(Path(sys.path[0]).parent))

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from numpy.linalg import inv, norm, svd, slogdet
from scipy.sparse.linalg import cg
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso, OrthogonalMatchingPursuit
from sklearn.preprocessing import StandardScaler

from src.utils import (
    corr_heatmap,
    block_thresholding,
    colnorm,
    convert_output,
    dct_matrix,
    n_atoms,
    proximal_logdet,
)
from wandb import Image


class SparseCoder:
    def __init__(
        self,
        X: np.ndarray,
        D: np.ndarray = None,
        method: str = 'lasso',
        params: dict[Any] = {},
    ):
        # ================================================================
        #                          Parameters
        # ================================================================
        defaults = {
            'explained_variance': None,
            'sparsity_level': None,
            'hard_thresh': False,
            'regularizer': 1e-3,
            'method': 'lasso',
            'momentum': None,
            # 'dict_type': None,
            'n_atoms': None,
            'max_iter': 1,
            'step': None,
            'tol': 1e-3,
        }
        defaults.update(params)
        # self.dict_type = defaults['dict_type']
        self.ev = defaults['explained_variance']
        self.thresh = defaults['hard_thresh']
        self.alpha = defaults['regularizer']
        self.max_iter = defaults['max_iter']
        self.momentum = defaults['momentum']
        self.method = defaults['method']
        self.step = defaults['step']
        self.tol = defaults['tol']

        # ================================================================
        #                          Variables
        # ================================================================
        self.X = X
        self.stalk_dim, self.n_examples = self.X.shape
        self.D = D
        self.n_atoms = (
            defaults['n_atoms'] if self.D is None else self.D.shape[1]
        )
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

        print(self.D.shape, self.S.shape)
        return None

    def _sparse_coding_lasso(
        self,
        S: np.ndarray,
        delta: np.ndarray = None,
    ) -> None:
        """
        Compute sparse codes via Lasso for multiple signals.
        Solves, for each column x_j of Y:
            minimize_S_i  0.5 * ||X_i - D @ S_i||_2^2  +  alpha * ||S_i^T||_2,1
        """
        if S is None:
            lasso = Lasso(
                alpha=self.alpha,
                fit_intercept=False,
                max_iter=self.max_iter,
                tol=self.tol,
            )
            lasso.fit(self.D, self.X)
            self.S = lasso.coef_.T
        else:
            grad = self.D.T @ (self.D @ S - self.X)
            self.delta = (
                -self.step * grad
                if self.momentum is None
                else -self.step * grad + self.momentum * delta
            )
            self.S = block_thresholding(
                (S + self.delta),
                self.alpha * self.step,
                hard=self.thresh,
            )

        return None

    def _sparse_coding_omp(self) -> None:
        """
        Compute sparse codes via Orthogonal Matching Pursuit (OMP) for multiple signals.
        Solves, for each column x_j of X:
            minimize_s ||x_j - D @ s||_2  subject to ||x_j - D @ s||_2 <= tol
        or stops after selecting up to max_iter atoms.
        """
        Domp = colnorm(self.D)
        omp = OrthogonalMatchingPursuit(
            n_nonzero_coefs=self.k0,
            tol=self.tol,
            fit_intercept=False,
            precompute=True,
        )
        omp.fit(Domp, self.X)
        self.S = omp.coef_.T

        return None

    @convert_output
    def fit(
        self,
        S: np.ndarray = None,
        delta: np.ndarray = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.method == 'pca':
            self._set_dictionary()
        elif self.method == 'lasso':
            self._sparse_coding_lasso(S=S, delta=delta)
        elif self.method == 'omp':
            self._sparse_coding_omp()
        else:
            raise ValueError(
                f'{self.method} is not a valid sparse coding method'
            )
        return self.D, self.S


class GlobalDict:
    def __init__(
        self,
        X: np.ndarray,
        agents: dict[int, dict[str, Any]],
        test_gt_dict: np.ndarray = None,
        run=None,
        n_nodes: int = 1,
        params: dict[Any] = {},
    ):
        # ================================================================
        #                          Parameters
        # ================================================================
        defaults = {
            'augmented_multiplier_sparse': None,
            'augmented_multiplier_dict': None,
            'dict_regularization_type': None,
            'explained_variance': None,
            'sparsity_tolerance': 1e-3,
            'sparse_regularizer': None,
            'dict_regularizer': None,
            'best_stepsize': False,
            'sparsity_level': None,
            'coding_method': 'lasso',
            'sparse_descent': False,
            'hard_thresh': False,
            'init_mode_dict': 'random',
            'init_mode_sparse': 'zeros',
            'dict_type': None,
            'momentum_D': None,
            'momentum_S': None,
            'n_atoms': None,
            'D_iters': None,
            'S_iters': 1,
            'patience': 0,
            'max_iter': 1,
            'use_CG': False,
            'split': None,
            'Dstep': None,
            'Sstep': None,
            'tol': 1e-3,
        }
        defaults.update(params)
        self.dict_regularization_type = defaults['dict_regularization_type']
        self.rho_sparse = defaults['augmented_multiplier_sparse']
        self.rho_dict = defaults['augmented_multiplier_dict']
        self.dict_regularizer = defaults['dict_regularizer']
        self.init_mode_sparse = defaults['init_mode_sparse']
        self.init_mode_dict = defaults['init_mode_dict']
        self.coding_method = defaults['coding_method']
        self.best_stepsize = defaults['best_stepsize']
        self.descent = defaults['sparse_descent']
        self.stol = defaults['sparsity_tolerance']
        self.ev = defaults['explained_variance']
        self.dict_type = defaults['dict_type']
        self.k0 = defaults['sparsity_level']
        self.alpha = defaults['sparse_regularizer']
        self.momentum_D = defaults['momentum_D']
        self.momentum_S = defaults['momentum_S']
        self.max_iter = defaults['max_iter']
        self.patience = defaults['patience']
        self.thresh = defaults['hard_thresh']
        self.n_atoms = defaults['n_atoms']
        self.S_iters = defaults['S_iters']
        self.D_iters = defaults['D_iters']
        self.use_CG = defaults['use_CG']
        self.Sstep = defaults['Sstep']
        self.Dstep = defaults['Dstep']
        self.split = defaults['split']
        self.tol = defaults['tol']
        self.stop_cond = 0
        self.global_step = 0
        self.loss = np.inf
        self.run = run

        # ================================================================
        #                          Variables
        # ================================================================
        self.X = X
        self.agents = agents
        self.stalk_dim, nN = self.X.shape
        self.const = 1 / norm(self.X, ord=2) ** 2
        self.n_examples = int(nN / n_nodes)
        self.n_nodes = n_nodes
        self.n_atoms = (
            self.stalk_dim
            if (self.n_atoms is None) or (self.dict_type == 'local_pca')
            else self.n_atoms
        )
        if self.dict_type == 'learnable':
            self.rho_sparse *= self.n_examples
            self.rho_dict *= self.n_examples
        self.D: np.ndarray = None
        self.localS: list[np.ndarray] = None
        self.localD: list[np.ndarray] = []
        self._init_dictionary()
        self._init_sparse()
        self.S: np.ndarray = np.hstack(self.localS)
        self.test_gt_dict = test_gt_dict

    def local_vars(
        self,
        varname: str,
        agent_idx: int,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        col = slice(
            agent_idx * self.n_examples, (agent_idx + 1) * self.n_examples
        )

        if varname == 'all':
            out = tuple(getattr(self, n)[:, col] for n in ('S', 'X', 'Z', 'U'))
        else:
            try:
                out = getattr(self, varname)[:, col]
            except AttributeError as exc:
                raise ValueError(
                    f"Unknown variable name '{varname}'."
                ) from exc
        return out

    def _init_dictionary(self) -> None:
        if self.init_mode_dict == 'random':
            self.D = np.random.randn(self.stalk_dim, self.n_atoms)
        elif self.init_mode_dict == 'fourier':
            assert self.n_atoms == self.stalk_dim, (
                'When using Fourier dictionary init, n_atoms must be equal to stalk_dim.'
            )
            self.D = dct_matrix(self.stalk_dim, norm='ortho')
        else:
            raise ValueError(
                f"Unknown initialization mode '{self.init_mode_dict}'."
            )
        return None

    def _init_sparse(self) -> None:
        if self.init_mode_sparse == 'zeros':
            self.localS = [
                np.zeros((self.n_atoms, self.n_examples))
                for _ in range(self.n_nodes)
            ]
        elif self.init_mode_sparse == 'random':
            self.localS = [
                np.random.randn(self.n_atoms, self.n_examples)
                for _ in range(self.n_nodes)
            ]
        elif self.init_mode_sparse == 'fourier':
            self.D = dct_matrix(self.stalk_dim, norm='ortho')
            self.S = self.D @ self.X
            self.localS = [
                self.local_vars(varname='S', agent_idx=i)
                for i in range(self.n_nodes)
            ]
        else:
            raise ValueError(
                f"Unknown initialization mode '{self.init_mode_sparse}'."
            )

    def row_sparsity_heatmap(self, avg: bool = False):
        local_norms = []
        agent_names = []
        for i in range(self.n_nodes):
            S_i = self.local_vars(varname='S', agent_idx=i)
            if avg:
                norms = norm(S_i, ord=2, axis=1) / self.n_examples
            else:
                norms = norm(S_i, ord=2, axis=1)
            local_norms.append(norms.reshape(-1, 1))
            agent_names.append(self.agents[i].model_name)
        local_norms = np.hstack(local_norms)
        return sns.heatmap(
            local_norms.T,
            yticklabels=agent_names,
            cmap='jet',
        ).set(xticks=[])

    def plot_dict(
        self,
        iteration: int,
    ) -> None:
        sns.heatmap(
            self.D,
            cmap='viridis',
            cbar_kws={'label': 'Value'},
        )
        plt.xlabel('')
        plt.ylabel('')
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        self.run.log(
            {f'global_dict_step{iteration}': Image(plt)}, step=self.global_step
        )
        plt.close()
        return None

    def eval(self) -> None:
        reg_var = self.Z if self.split in ['both', 'sparse'] else self.S
        sparse_reg = (
            np.count_nonzero(norm(reg_var, ord=2, axis=1)) * self.alpha
            if self.thresh
            else self.alpha * norm(reg_var, ord=2, axis=1).sum()
        )
        if self.dict_regularization_type == 'logdet':
            sign, det = slogdet(self.D.T @ self.D)
            dict_reg = sign * det
        else:
            dict_reg = 0
        mse = (norm(self.X - self.D @ self.S) ** 2) / 2
        loss = mse + sparse_reg + dict_reg
        norm_loss = self.const * loss
        if (self.loss - loss) < self.tol:
            self.stop_cond += 1
        self.loss = loss
        self.nmse = self.const * mse
        self.sparsity = np.count_nonzero(norm(self.S, ord=2, axis=1))
        if self.run is not None:
            if self.test_gt_dict is not None:
                approx_error = (norm(self.D - self.test_gt_dict) ** 2) / (
                    norm(self.test_gt_dict) ** 2
                )
                self.run.log(
                    {'Dict approx error': approx_error},
                    step=self.global_step,
                )
                self.run.log(
                    {'Dict_residual': self.residual},
                    step=self.global_step,
                )
            # self.run.log({'Total loss': self.loss}, step=self.global_step)
            self.run.log({'Norm. tot. loss': norm_loss}, step=self.global_step)
            self.run.log({'NMSE': self.nmse}, step=self.global_step)
            self.global_step += 1

        return None

    def compute_stepsize(
        self,
        varname: str,
        splitted: bool = True,
    ) -> None:
        if varname == 'D':
            self.Dstep = (
                1.0 / norm(self.S, 2) ** 2 + self.rho_sparse
                if splitted
                else self.Dstep
            )
        elif varname == 'S':
            pass
        else:
            raise ValueError(f"Unknown variable name '{varname}'.")

    def _sparse_coding(
        self,
        coder_params: dict[Any] = {},
    ) -> None:
        self.localS = []
        delta = 0
        for _ in range(self.S_iters):
            for i in range(self.n_nodes):
                X_i = self.local_vars(varname='X', agent_idx=i)
                coder = SparseCoder(
                    X_i,
                    D=self.D,
                    params=coder_params,
                )
                S_i = (
                    self.local_vars(varname='S', agent_idx=i)
                    if self.descent
                    else None
                )
                D_i, S_i = coder.fit(S=S_i, delta=delta, out='numpy')
                delta = coder.delta if self.descent else 0
                self.localS.append(S_i)
                if self.dict_type == 'local_pca':
                    self.localD.append(D_i)
                # else:
                #     self.S[
                #         :, i * self.n_examples : (i + 1) * self.n_examples
                #     ] = S_i
                #     self.eval()
            if self.stop_cond >= self.patience:
                break
        self.S = np.hstack(self.localS)
        self.eval()
        return None

    def _splitted_sparse_coding(self) -> None:
        self.localS = []
        localZ = []
        localU = []
        A = inv(self.D.T @ self.D + self.rho_sparse * np.eye(self.n_atoms))
        for i in range(self.n_nodes):
            S_i, X_i, Z_i, U_i = self.local_vars(varname='all', agent_idx=i)
            S_i = A @ (self.D.T @ X_i + self.rho_sparse * (Z_i - U_i))
            Z_i = block_thresholding(
                (S_i + U_i),
                self.alpha / self.rho_sparse,
                hard=self.thresh,
            )
            U_i += S_i - Z_i
            self.localS.append(S_i)
            localZ.append(Z_i)
            localU.append(U_i)
        self.S = np.hstack(self.localS)
        self.Z = np.hstack(localZ)
        self.U = np.hstack(localU)
        self.eval()
        return None

    def _update_dictionary(self) -> None:
        # self.S = np.hstack(self.localS)

        if self.D_iters is None:
            self.D = np.linalg.lstsq(
                self.S @ self.S.T, self.S @ self.X.T, rcond=None
            )[0].T
            self.D = colnorm(self.D)
            self.eval()
        else:
            delta_D = None if self.momentum_D is None else 0
            for _ in range(self.D_iters):
                grad = (self.D @ self.S - self.X) @ self.S.T
                delta_D = (
                    -self.Dstep * grad
                    if self.momentum_D is None
                    else -self.Dstep * grad + self.momentum_D * delta_D
                )
                self.D += delta_D
                self.D = colnorm(self.D)
                self.eval()
                if self.stop_cond >= self.patience:
                    break
        return None

    def cg_D_update(
        self,
        tol=1e-5,
        maxit=200,
    ) -> None:
        A = self.S @ self.S.T + self.rho_dict * np.eye(self.n_atoms)
        B = self.X @ self.S.T + self.rho_dict * (self.P - self.R)
        for r in range(B.shape[0]):
            self.D[r], _ = cg(A, B[r], rtol=tol, maxiter=maxit)
        return None

    def _splitted_update_dictionary(self) -> None:
        # Closed-form solution
        if self.D_iters is None:
            D_old = self.D.copy()
            if self.dict_regularization_type is None:
                self.D = (
                    self.X @ self.S.T + self.rho_dict * (self.P - self.R)
                ) @ inv(
                    self.S @ self.S.T + self.rho_dict * np.eye(self.n_atoms)
                )
            elif self.dict_regularization_type == 'logdet':
                assert self.Dstep is not None, (
                    'step size in D cannot be None if "logdet" regularization is used'
                )
                linearized_reg = (
                    self.dict_regularizer * self.D @ inv(self.D.T @ self.D)
                )
                D_new = (
                    self.X @ self.S.T
                    + self.rho_dict * (self.P - self.R)
                    + linearized_reg
                ) @ inv(
                    self.S @ self.S.T + self.rho_dict * np.eye(self.n_atoms)
                )
                self.D = self.D + self.Dstep * (D_new - self.D)
                # Q, unconstrained_evals, Wt = svd(self.D)
                # constrained_evals = proximal_logdet(
                #     unconstrained_evals,
                #     epsilon=0,
                #     lam=self.dict_regularizer,
                # )
                # self.D = Q @ np.diag(constrained_evals) @ Wt
            else:
                pass

            self.residual = norm(self.D - D_old) ** 2
            self.P = colnorm(self.D + self.R)
            self.R += self.D - self.P
            self.eval()

        # Smooth descending
        else:
            if self.best_stepsize:
                self.compute_stepsize(varname='D', splitted=True)
            for _ in range(self.D_iters):
                if self.use_CG:
                    self.cg_D_update()
                else:
                    grad = (
                        self.D @ self.S - self.X
                    ) @ self.S.T + self.rho_dict * (self.D - self.P + self.R)
                    self.D -= self.Dstep * grad
                self.P = colnorm(self.D + self.R)
                self.R += self.D - self.P
                self.eval()
                if self.stop_cond >= self.patience:
                    break
        return None

    @convert_output
    def fit(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Fit a global dictionary D and the sparse codes S_i, solving:
            min_{D,S_i}  0.5 * ||X - D @ S||_F^2  +  lambda * sum_i ||S_i^T||_{2,1}
        """

        good_iters = 0
        coder_params = {
            'method': self.coding_method,
            'explained_variance': self.ev,
            'momentum': self.momentum_S,
            'sparsity_level': self.k0,
            'sparsity_tolerance': self.stol,
            'hard_thresh': self.thresh,
            'regularizer': self.alpha,
            'max_iter': self.max_iter,
            'n_atoms': self.n_atoms,
            'step': self.Sstep,
            'tol': self.tol,
        }

        if self.dict_type == 'learnable':
            if self.split in ['both', 'dict']:
                self.P = np.zeros_like(self.D)
                self.R = np.zeros_like(self.D)
            if self.split in ['both', 'sparse']:
                self.Z = np.zeros_like(self.S)
                self.U = np.zeros_like(self.S)

            for i in range(self.max_iter):
                # print(f'Global iteration {i + 1}/{self.max_iter}')
                # =============================================================
                #                    UPDATE GLOBAL DICTIONARY
                # =============================================================
                if self.split in ['both', 'dict']:
                    self._splitted_update_dictionary()
                else:
                    self._update_dictionary()
                # =============================================================
                #              UPDATE LOCAL SPARSE REPRESENTATIONS
                # =============================================================
                if self.split in ['both', 'sparse']:
                    self._splitted_sparse_coding()
                else:
                    self._sparse_coding(coder_params=coder_params)

                # if i % 100 == 0:
                # corr_heatmap(self.D)
                # self.run.log(
                #     {f'dict_corr_step{i}': Image(plt)},
                #     step=self.global_step,
                # )
                # plt.close()
                # self.plot_dict(iteration=i)

                # =============================================================
                #                  EARLY STOPPING CRITERION
                # =============================================================
                if self.stop_cond >= self.patience:
                    print(
                        f'Converged after {i + 1} iterations with tol={self.tol}. Loss: {self.loss}.'
                    )
                    break
                else:
                    good_iters += 1

                if good_iters >= 20:
                    self.stop_cond = 0

            self.row_sparsity_heatmap()
            plt.tight_layout()
            # self.run.summary['agents_sparsity'] = Image(plt)
            self.run.log({'sparsity': Image(plt)}, step=self.global_step)
            plt.clf()
            plt.cla()

        # elif self.dict_type == 'learnable-sparse-splitted':
        #     # Init dual and splitting variables
        #     self.U = np.zeros_like(self.S)
        #     self.Z = np.zeros_like(self.S)
        #     self.P = np.zeros_like(self.D)
        #     self.R = np.zeros_like(self.D)
        #     for i in range(self.max_iter):
        #         self._splitted_update_dictionary(splitted_reg=False)
        #         self._splitted_sparse_coding()
        #         # self._sparse_coding(gd=True, coder_params=coder_params)

        #         if self.stop_cond >= self.patience:
        #             print(
        #                 f'Converged after {i + 1} iterations with tol={self.tol}. Loss: {self.loss}.'
        #             )
        #             break
        #         else:
        #             self.stop_cond = 0

        elif self.dict_type == 'fourier':
            self.D = dct_matrix(self.stalk_dim)
        elif self.dict_type == 'local_pca':
            assert self.S_iters == 1, (
                'S_iters must be 1 for local PCA dictionary.'
            )
            self.D = None
            coder_params['method'] = 'pca'
            self._sparse_coding(coder_params=coder_params)
        else:
            raise ValueError(f"Unknown dictionary type '{self.dict_type}'.")

        if self.localD == []:
            self.localD = [self.D for _ in range(self.n_nodes)]

        return self.localD, self.localS

    def return_metrics(self) -> dict[str, Any]:
        """Return metrics of the global dictionary."""

        metrics = {'agent_id': [], 'sparsity': [], 'nmse': []}
        for i in range(self.n_nodes):
            # S_i = self.local_vars(varname='S', agent_idx=i)
            X_i = self.local_vars(varname='X', agent_idx=i)
            S_i = self.localS[i]
            D_i = self.localD[i] if self.dict_type == 'local_pca' else self.D
            metrics['agent_id'].append(i)
            metrics['sparsity'].append(n_atoms(torch.from_numpy(S_i)))
            metrics['nmse'].append(
                (norm(X_i - D_i @ S_i) ** 2 / 2) / norm(X_i) ** 2
            )

        return metrics
