""""""

import sys
from pathlib import Path

sys.path.append(str(Path(sys.path[0]).parent))

from numpy.linalg import svd, norm
from tqdm.auto import tqdm
from typing import Any

from src.sheaf import Edge, Network


class EdgeProblem:
    """"""

    def __init__(
        self,
        edge: Edge,
        params: dict[Any],
        run=None,
    ):
        # ================================================================
        #                          Variables
        # ================================================================
        self.edge = edge
        self.F_i, self.F_j = self.edge.get_restriction_maps()
        self.X_i, self.X_j, _, _ = self.edge.get_latents(
            prewhite=False,
            scale=True,
            out='numpy',
        )
        self.S_i, self.S_j = self.edge.get_sparse_representations(out='numpy')
        self.cov_i, self.cov_j, self.cov_ij, self.cov_ji = (
            self.edge.get_latent_covariances(out='numpy')
        )
        self.Z_ij = self.edge.get_mask(out='numpy')
        self.D_i, self.D_j = self.edge.get_dictionaries(out='numpy')
        self.N = self.X_i.shape[1]
        self.d = self.D_i.shape[0]

        # ================================================================
        #                          Parameters
        # ================================================================
        defaults = {
            'n_iters': 1,
            'sparse_regularizer_head': None,  # gamma
            'sparse_regularizer_tail': None,  # gamma
            'lagrange_multiplier': None,  # alpha
            'mask_regularizer': None,  # lambda_
            'align_comm_tradeoff': None,  # beta
            'proximal_stepsize': None,  # mu
            'verbose': False,
        }
        defaults.update(params)
        self.gamma1 = defaults['sparse_regularizer_head']
        self.gamma2 = defaults['sparse_regularizer_tail']
        self.alpha = defaults['lagrange_multiplier']
        self.lambda_ = defaults['mask_regularizer']
        self.beta = defaults['align_comm_tradeoff']
        # self.ev = defaults['explained_variance']
        self.mu = defaults['proximal_stepsize']
        self.n_iters = defaults['n_iters']
        self.run = run

    # ================================================================
    #                   Possible Evaluation Metrics
    # ================================================================

    def _semantic_misalignment(
        self,
        sub_projection: bool = True,
        normalize: bool = True,
    ):
        if sub_projection:
            sm = norm(
                self.F_j @ self.D_j @ self.S_j - self.D_i @ self.S_i, ord='fro'
            )
        else:
            sm = norm(self.F_j @ self.X_j - self.X_i, ord='fro')

        if normalize:
            sm /= norm(self.D_i @ self.S_i, ord='fro')
        return sm

    def _sparse_reconstructions(self):
        return norm(self.X_j - self.D_j @ self.S_j, ord='fro') + norm(
            self.X_i - self.D_i @ self.S_i, ord='fro'
        )

    def _sparse_regularizers(self):
        return (
            self.gamma1 * norm(self.S_i, ord=2, axis=1).sum()
            + self.gamma2 * norm(self.S_j, ord=2, axis=1).sum()
        )

    # ================================================================
    #                          Edge Update
    # ================================================================

    def update_edge(self):
        self.edge.update_restriction_maps(self.F_i, self.F_j)
        self.edge.update_sparse_representations(self.S_i, self.S_j)
        self.edge.update_mask(self.Z_ij)
        self.edge.update_alignment_loss(self.loss)

    # ================================================================
    #                      Alignment Evaluation
    # ================================================================

    def eval(self) -> float:
        self.loss = self._semantic_misalignment()
        if self.run is not None:
            self.run.log({f'Loss (edge {self.edge.id})': self.loss})
        return self.loss

    # ================================================================
    #                 Fitting the Alignment Process
    # ================================================================
    def fit(self):
        """Update restriction maps solving orthogonal Procrustes problem"""
        U, _, Vt = svd((self.cov_j @ self.cov_i.T) / self.N)
        self.F_j = (U @ Vt).T
        self.eval()
        self.update_edge()
        return None

    # def eval(self) -> float:
    #     mis = self._semantic_misalignment()
    #     rec = self._sparse_reconstructions()
    #     reg = self._sparse_regularizers()
    #     self.loss = mis + rec + reg
    #     if self.run is not None:
    #         self.run.log(
    #             {
    #                 f'Training Loss edge {self.edge.id} - ev {self.ev}': self.loss
    #             }
    #         )
    #     return self.loss

    # def fit(
    #     self,
    # ) -> None:
    #     for _ in range(self.n_iters):
    #         # Update S_j
    #         A_j = np.vstack([self.F_j @ self.D, self.D])
    #         C_j = np.vstack([self.D @ self.S_i, self.X_j])
    #         self.S_j = block_st(
    #             self.S_j - 2 * self.mu * A_j.T @ (A_j @ self.S_j - C_j),
    #             self.mu * self.gamma1,
    #         )
    #         # Update S_i
    #         A_i = np.vstack([self.D, self.D])
    #         C_i = np.vstack([self.F_j @ self.D @ self.S_j, self.X_i])
    #         self.S_i = block_st(
    #             self.S_i - 2 * self.mu * A_i.T @ (A_i @ self.S_i - C_i),
    #             self.mu * self.gamma2,
    #         )

    #         # Update F_ji solving orthogonal Procrustes problem
    #         U, _, V = svd(self.S_j @ self.S_i.T)
    #         self.F_j = U @ V.T

    #         self.eval()

    #     self.update_edge()

    #     return None


class EdgeProblemProcrustes(EdgeProblem):
    def __init__(
        self,
        edge: Edge,
        params: dict[Any],
        run=None,
    ):
        super().__init__(
            edge=edge,
            params=params,
            run=run,
        )

    def eval(self) -> float:
        self.loss = self._semantic_misalignment()
        if self.run is not None:
            self.run.log({f'Loss (edge {self.edge.id})': self.loss})
        return self.loss

    def fit(self):
        """Update restriction maps solving orthogonal Procrustes problem
        on a common subspace.
        """
        U, _, Vt = svd(
            self.D_j @ (self.S_j @ self.S_i.T) / self.N @ self.D_i.T
        )
        self.F_j = (U @ Vt).T
        self.eval()
        self.update_edge()
        return None


def main():
    print('Start performing sanity tests...')

    agents = {
        0: {
            'model': 'vit_small_patch16_224',
            'dataset': 'cifar10',
            'seed': 42,
        },
        1: {
            'model': 'vit_small_patch32_224',
            'dataset': 'cifar10',
            'seed': 42,
        },
    }
    coder_params = {
        'explained_variance': 0.9,
        'sparsity_level': 0.1,
        'regularizer': 1e-3,
        'n_atoms': None,
        'dictionary_type': 'fourier',
        'max_iter': 100,
        'tol': 1e-2,
    }
    alignment_params = {}

    # =============================================================
    #                      BUILD THE NETWORK
    # =============================================================
    print('Preparing the network...')
    net = Network(
        agents_info=agents,
        coder_params=coder_params,
        device='cpu',
    )
    print('[Passed]')

    # =============================================================
    #             TEST ALL THE ALIGNMENT FORMULATIONS
    # =============================================================
    print('Aligning the latent spaces...')
    for edge in tqdm(list(net.edges.values())):
        print(f'Edge {edge.id} alignment...', end='\t')
        prob = EdgeProblemProcrustes(
            edge=edge,
            params=alignment_params,
        )
        prob.fit()
        print('[Passed]')
    print('...all latent spaces aligned!')

    return None


if __name__ == '__main__':
    main()
