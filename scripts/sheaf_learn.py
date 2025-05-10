"""
This python module handles the network formation using numpy arrays.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(sys.path[0]).parent))

from src.linear import Network
from utils import memoize
from typing import Any
from tqdm.auto import tqdm
import torch
from scipy.linalg import solve_sylvester, polar
import numpy as np
import hydra


def generate_data(cfg):
    """Generate data using NumPy arrays."""
    agents_info: dict[int, dict[str, Any]] = {}
    k = 0  # assumes single model type, as before
    for i in range(cfg.network.n_agents):
        agents_info[i] = {
            'model': cfg.network.models[k],
            'dataset': cfg.network.datasets[k],
        }

    # Note: assumes Network has been updated to work with NumPy arrays
    net = Network(agents_info=agents_info)
    net.graph_initialization(p=cfg.network.p_edges)
    net.data_initialization()
    return net


def block_st(
    X: np.ndarray,
    beta: float = 0.1,
) -> np.ndarray:
    """
    Apply block soft thresholding operator to each column of a matrix.
    NumPy equivalent of the original Torch version.
    """
    norms = np.linalg.norm(X, axis=0)
    scales = np.maximum(1 - beta / norms, 0.0)
    return X * scales


@memoize
def sheaf_learn(
    cfg,
    net: Network,
) -> None:
    """Learning the sheaf restriction maps and set of edges using NumPy."""
    n_iter = cfg.algorithm.n_iter
    alpha = cfg.algorithm.alpha
    lambda_ = cfg.algorithm.lambda_
    tau_a = cfg.algorithm.tau_a
    tau_r = cfg.algorithm.tau_r
    n_edges = cfg.algorithm.n_edges
    verbose = cfg.algorithm.verbose

    for i, j in tqdm(net.graph.get_edgelist()):
        F_ij = net.agents[i].restriction_maps[j].detach().cpu().numpy()
        F_ji = net.agents[j].restriction_maps[i].detach().cpu().numpy()
        n_i = F_ij.shape[1]
        n_j = F_ji.shape[1]
        n_ij = F_ij.shape[0]

        S_i, S_j, S_ij, S_ji = net.get_sample_covs((i, j), out='np')

        Y_i = np.random.randn(n_i, n_ij)
        Y_j = np.random.randn(n_j, n_ij)
        U_i = np.random.randn(n_i, n_ij)
        U_j = np.random.randn(n_j, n_ij)
        V_ij = np.random.randn(n_i + n_j, n_ij)
        U_ij = np.random.randn(n_i + n_j, n_ij)

        for k in range(n_iter):
            # Update F_ji via Sylvester
            A_j = np.eye(n_ij) - F_ij @ F_ij.T
            B_j = F_ij @ S_i @ F_ij.T + alpha * np.eye(n_ij)
            C_j = (
                3 * F_ij @ S_ij
                + (alpha / 2) * (Y_j + V_ij[n_i:, :] - U_j - U_ij[n_i:, :]).T
            )
            B_inv = np.linalg.inv(B_j)
            S_inv = np.linalg.inv(S_j)
            F_ji = solve_sylvester(B_inv @ A_j, S_inv, B_inv @ C_j @ S_inv)

            # Update F_ij via Sylvester
            A_i = np.eye(n_ij) - F_ji @ F_ji.T
            B_i = F_ji @ S_j @ F_ji.T + alpha * np.eye(n_ij)
            C_i = (
                3 * F_ji @ S_ji
                + (alpha / 2) * (Y_i + V_ij[:n_i, :] - U_i - U_ij[:n_i, :]).T
            )
            B_inv = np.linalg.inv(B_i)
            S_inv = np.linalg.inv(S_i)
            F_ij = solve_sylvester(B_inv @ A_i, S_inv, B_inv @ C_i @ S_inv)

            # Store old values for convergence check
            Y_i_prev = Y_i.copy()
            Y_j_prev = Y_j.copy()
            V_ij_prev = V_ij.copy()

            # Update dual variables
            Y_i = polar(F_ij.T + U_i)[0]
            Y_j = polar(F_ji.T + U_j)[0]
            V_ij = block_st(
                np.vstack((F_ij.T, F_ji.T)) + U_ij, beta=lambda_ / alpha
            )

            # Dual residuals
            D_i = alpha * (Y_i - Y_i_prev)
            D_j = alpha * (Y_j - Y_j_prev)
            D_ij = alpha * (V_ij - V_ij_prev)

            # Primal residuals
            R_i = F_ij.T - Y_i
            R_j = F_ji.T - Y_j
            R_ij = np.vstack((F_ij.T, F_ji.T)) - V_ij

            # Update auxiliary variables
            U_i += R_i
            U_j += R_j
            U_ij += R_ij

            # Check stopping criterion
            conditions = []
            conditions.append(
                np.linalg.norm(R_i, ord='fro')
                < tau_a * np.sqrt(n_ij * n_i)
                + tau_r
                * np.max(
                    [
                        np.linalg.norm(Y_i, ord='fro'),
                        np.linalg.norm(F_ij, ord='fro'),
                    ]
                )
            )
            conditions.append(
                np.linalg.norm(R_j, ord='fro')
                < tau_a * np.sqrt(n_ij * n_j)
                + tau_r
                * np.max(
                    [
                        np.linalg.norm(Y_j, ord='fro'),
                        np.linalg.norm(F_ji, ord='fro'),
                    ]
                )
            )
            conditions.append(
                np.linalg.norm(R_ij, ord='fro')
                < tau_a * np.sqrt(n_ij * (n_i + n_j))
                + tau_r
                * np.max(
                    [
                        np.linalg.norm(V_ij, ord='fro'),
                        np.linalg.norm(np.vstack((F_ij.T, F_ji.T)), ord='fro'),
                    ],
                )
            )
            conditions.append(
                np.linalg.norm(D_i, ord='fro')
                < tau_a * np.sqrt(n_ij * n_i)
                + tau_r * alpha * np.linalg.norm(U_i, ord='fro')
            )
            conditions.append(
                np.linalg.norm(D_j, ord='fro')
                < tau_a * np.sqrt(n_ij * n_j)
                + tau_r * alpha * np.linalg.norm(U_j, ord='fro')
            )
            conditions.append(
                np.linalg.norm(D_ij, ord='fro')
                < tau_a * np.sqrt(n_ij * (n_i + n_j))
                + tau_r * alpha * np.linalg.norm(U_ij, ord='fro')
            )

            if verbose:
                if all(conditions):
                    print(
                        f'Early Stopping \n Convergence reached for edge ({i}, {j}) after {k} iterations.'
                    )
                    break

                if k == n_iter - 1:
                    print(f'Max iterations reached for edge ({i}, {j}).')

        net.update_restriction_maps(
            (i, j),
            torch.from_numpy(F_ij.astype(np.float32)),
            torch.from_numpy(F_ji.astype(np.float32)),
        )

    net.update_graph(n_edges=n_edges, lambda_=lambda_)

    return net


@hydra.main(
    config_path='../.conf/hydra/network',
    config_name='sheaf_learn',
    version_base='1.3',
)
def main(cfg) -> None:
    """The main script loop."""

    print('Starting the data generation...')
    net = generate_data(cfg)
    print('Data generation completed.')

    print('Starting the sheaf learning...')
    net = sheaf_learn(cfg, net)

    # for i, j in tqdm(net.graph.get_edgelist()):
    #     F_ij = net.agents[i].restriction_maps[j].to(torch.float32)
    #     F_ji = net.agents[j].restriction_maps[i].to(torch.float32)
    #     net.update_restriction_maps((i, j), F_ij, F_ji)

    print('Sheaf learning completed.')

    print('Starting the evaluation...')
    net.eval(
        dataset=cfg.evaluation.dataset,
        seed=cfg.seed,
    )

    return None


if __name__ == '__main__':
    main()
