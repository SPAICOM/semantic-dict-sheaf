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
        run: None,
    ):
        # ================================================================
        #                          Variables
        # ================================================================
        self.edge = edge
        self.F_i, self.F_j = self.edge.get_restriction_maps()
        self.X_i, self.X_j, _, _ = self.edge.get_latents(
            prewhite=False,
            scale=False,
            out='numpy',
        )
        self.S_i, self.S_j = self.edge.get_sparse_representations(out='numpy')
        self.cov_i, self.cov_j, self.cov_ij, self.cov_ji = (
            self.edge.get_latent_covariances(out='numpy')
        )
        self.Z_ij = self.edge.get_mask(out='numpy')
        self.D_i, self.D_j = self.edge.get_dictionaries(out='numpy')
        self.N = self.X_i.shape[1]
        self.d = None if self.D_i is None else self.D_i.shape[0]

        # ================================================================
        #                          Parameters
        # ================================================================
        self.params = {
            'projected': True,
        }
        self.params.update(params)
        self.run = run

    # ================================================================
    #                   Possible Evaluation Metrics
    # ================================================================

    def _semantic_misalignment(
        self,
        normalize: bool = True,
    ):
        if self.params['projected']:
            sm = norm(
                self.F_j @ self.D_j @ self.S_j - self.D_i @ self.S_i, ord='fro'
            )
        else:
            sm = norm(self.F_j @ self.X_j - self.X_i, ord='fro')

        if normalize:
            normalizer = (
                norm(self.D_i @ self.S_i, ord='fro')
                if self.params['projected']
                else norm(self.X_i, ord='fro')
            )
            # m, n = self.X_i.shape
            # sm /= m * n
            sm /= normalizer
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
    # def fit(self):
    #     """Update restriction maps solving orthogonal Procrustes problem"""
    #     U, _, Vt = svd((self.cov_j @ self.cov_i.T) / self.N)
    #     self.F_j = (U @ Vt).T
    #     self.eval()
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

    # ================================================================
    #                 Fitting the Alignment Process
    # ================================================================
    def fit(self):
        """Update restriction maps solving orthogonal Procrustes problem
        on a common subspace.
        """
        U, _, Vt = (
            svd(self.D_j @ (self.S_j @ self.S_i.T) / self.N @ self.D_i.T)
            if self.params['projected']
            else svd((self.X_j @ self.X_i.T) / self.N)
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
