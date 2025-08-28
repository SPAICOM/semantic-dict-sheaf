"""
This python module handles the network formation using numpy arrays.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(sys.path[0]).parent))

from src.sheaf import Network
from src.utils import projected_proximal  ### Not good
from utils import memoize
from typing import Any
from tqdm.auto import tqdm
import torch
from scipy.linalg import solve_sylvester, polar
from numpy.linalg import svd
import numpy as np
import hydra


def generate_data(cfg):
    """Generate data using NumPy arrays."""
    agents_info: dict[int, dict[str, Any]] = {}
    models = cfg.network.models
    for i, model in enumerate(list(models.keys())):
        agents_info[i] = {
            'model': model,
            'dataset': models[model][0],
            'seed': models[model][1],
        }

    net = Network(agents_info=agents_info)
    net.graph_initialization(
        p=cfg.network.p_edges,
        mask_restrictions=cfg.network.mask_restrictions,
    )
    return net


@memoize
def sheaf_learn(
    cfg,
    net: Network,
) -> None:
    """Learning the sheaf restriction maps and set of edges using NumPy."""
    n_iter = cfg.algorithm.n_iter
    alpha = cfg.algorithm.alpha
    lambda_ = cfg.algorithm.lambda_
    beta = cfg.algorithm.beta
    mu = cfg.algorithm.mu
    tau_a = cfg.algorithm.tau_a
    tau_r = cfg.algorithm.tau_r
    n_edges = cfg.algorithm.n_edges
    verbose = cfg.algorithm.verbose

    for i, j in tqdm(net.graph.get_edgelist()):
        F_ij = net.agents[i].restriction_maps[j].detach().cpu().numpy()
        F_ji = net.agents[j].restriction_maps[i].detach().cpu().numpy()
        S_t = (
            net.edge_masks[(i, j)].detach().cpu().numpy()
            if cfg.network.mask_restrictions
            else None
        )
        n_i = F_ij.shape[1]
        n_j = F_ji.shape[1]
        n_ij = F_ij.shape[0]

        S_i, S_j, S_ij, S_ji = net.get_sample_covs((i, j), out='np')

        if net.is_connection_graph:
            # Update F_ji solving orthogonal Procrustes problem
            U, _, V = svd(S_ji)
            F_ji = U @ V.T
        else:
            for k in range(n_iter):
                Y_i = np.random.randn(n_i, n_ij)
                Y_j = np.random.randn(n_j, n_ij)
                U_i = np.random.randn(n_i, n_ij)
                U_j = np.random.randn(n_j, n_ij)

                # Update F_ji via Sylvester
                if S_t is None:
                    A_j = beta * np.eye(n_ij) - (1 - beta) * F_ij @ F_ij.T
                    B_j = (1 - beta) * (
                        F_ij @ F_ij.T + alpha / 2 * np.eye(n_ij)
                    )
                    C_j = (1 - beta) * (3 * F_ij @ S_ij) + (alpha / 2) * (
                        Y_j - U_j
                    ).T
                else:
                    A_j = S_t @ (
                        beta * np.eye(n_ij) - (1 - beta) * F_ij @ F_ij.T
                    )
                    B_j = (1 - beta) * (
                        S_t @ F_ij @ S_i @ F_ij.T @ S_t
                        + alpha / 2 * np.eye(n_ij)
                    )
                    C_j = (1 - beta) * S_t @ (3 * F_ij @ S_ij) + (
                        alpha / 2
                    ) * (Y_j - U_j).T
                B_inv = np.linalg.inv(B_j)
                S_inv = np.linalg.inv(S_j)
                F_ji = solve_sylvester(B_inv @ A_j, S_inv, B_inv @ C_j @ S_inv)

                # Update F_ij via Sylvester
                if S_t is None:
                    A_i = beta * np.eye(n_ij) - (1 - beta) * F_ji @ F_ji.T
                    B_i = (1 - beta) * (
                        F_ji @ S_j @ F_ji.T + alpha / 2 * np.eye(n_ij)
                    )
                    C_i = (1 - beta) * (3 * F_ji @ S_ji) + (alpha / 2) * (
                        Y_i - U_i
                    ).T
                else:
                    A_i = S_t @ (
                        beta * np.eye(n_ij) - (1 - beta) * F_ji @ F_ji.T
                    )
                    B_i = (1 - beta) * (
                        S_t @ F_ji @ S_j @ F_ji.T @ S_t
                        + alpha / 2 * np.eye(n_ij)
                    )
                    C_i = (1 - beta) * S_t @ (3 * F_ji @ S_ji) + (
                        alpha / 2
                    ) * (Y_i - U_i).T
                B_inv = np.linalg.inv(B_i)
                S_inv = np.linalg.inv(S_i)
                F_ij = solve_sylvester(B_inv @ A_i, S_inv, B_inv @ C_i @ S_inv)

                # Store old values for convergence check
                Y_i_prev = Y_i.copy()
                Y_j_prev = Y_j.copy()

                # Update dual variables
                Y_i = polar(F_ij.T + U_i)[0]
                Y_j = polar(F_ji.T + U_j)[0]

                # Update edge restriction map mask
                if S_t is not None:
                    S_t = projected_proximal(
                        S_t
                        - mu
                        * net.edge_loss_grad(
                            edge=(i, j),
                            beta=beta,
                            autodiff=True,
                        ),
                        lambda_=lambda_,
                    )

                # Dual residuals
                D_i = alpha * (Y_i - Y_i_prev)
                D_j = alpha * (Y_j - Y_j_prev)

                # Primal residuals
                R_i = F_ij.T - Y_i
                R_j = F_ji.T - Y_j

                # Update auxiliary variables
                U_i += R_i
                U_j += R_j

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
                    np.linalg.norm(D_i, ord='fro')
                    < tau_a * np.sqrt(n_ij * n_i)
                    + tau_r * alpha * np.linalg.norm(U_i, ord='fro')
                )
                conditions.append(
                    np.linalg.norm(D_j, ord='fro')
                    < tau_a * np.sqrt(n_ij * n_j)
                    + tau_r * alpha * np.linalg.norm(U_j, ord='fro')
                )

                if verbose:
                    if all(conditions):
                        print(
                            f'Early Stopping \n Convergence reached for edge ({i}, {j}) after {k} iterations.'
                        )
                        break

                    if k == n_iter - 1:
                        print(f'Max iterations reached for edge ({i}, {j}).')

        S_t = None if S_t is None else torch.from_numpy(S_t.astype(np.float32))
        net.update_restriction_maps(
            (i, j),
            torch.from_numpy(F_ij.astype(np.float32)),
            torch.from_numpy(F_ji.astype(np.float32)),
            S_t,
        )

    net.update_graph(n_edges=n_edges, beta=beta, lambda_=lambda_)

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
    print('Sheaf learning completed.')

    print('Starting the evaluation...')
    net.eval(
        dataset=cfg.evaluation.dataset,
        seed=cfg.seed,
    )

    return None


if __name__ == '__main__':
    main()
