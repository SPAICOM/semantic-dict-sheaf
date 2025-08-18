"""
This Python module handles the network formation and sheaf learning using JAX.
It replaces NumPy/SciPy routines with their JAX equivalents, jit-compiles the
per-edge ADMM solver, and accelerates computation via XLA.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(sys.path[0]).parent))

from src.sheaf import Network
from typing import Any
import numpy as np
import jax.numpy as jnp
from jax import random, jit, lax
import hydra


def generate_data(cfg) -> Network:
    """Generate data using NumPy arrays (unchanged)."""
    agents_info: dict[int, dict[str, Any]] = {}
    k = 0  # assumes single model type
    for i in range(cfg.network.n_agents):
        agents_info[i] = {
            'model': cfg.network.models[k],
            'stalk_dim': cfg.network.node_dims[k],
        }
    net = Network(agents_info=agents_info)
    net.graph_initialization(p=cfg.network.p_edges)
    net.edges_capacity()
    net.data_initialization()
    return net


@jit
def block_st(X: jnp.ndarray, beta: float) -> jnp.ndarray:
    """
    Block soft-thresholding operator on columns of X.
    """
    norms = jnp.linalg.norm(X, axis=0)
    scales = jnp.maximum(1 - beta / norms, 0.0)
    return X * scales


def init_edge_state(
    net: Network,
    edge: tuple[int, int],
    key: jnp.ndarray,
) -> tuple:
    """
    Initialize all per-edge ADMM variables and pack into a state tuple.
    """
    i, j = edge
    F_ij = jnp.array(net.agents[i].restriction_maps[j].detach().cpu().numpy())
    F_ji = jnp.array(net.agents[j].restriction_maps[i].detach().cpu().numpy())
    S_i, S_j, S_ij, S_ji = net.get_sample_covs((i, j), out='jax')
    n_i = F_ij.shape[1]
    n_j = F_ji.shape[1]
    n_ij = F_ij.shape[0]
    keys = random.split(key, 6)
    Y_i = random.normal(keys[0], (n_i, n_ij))
    Y_j = random.normal(keys[1], (n_j, n_ij))
    U_i = random.normal(keys[2], (n_i, n_ij))
    U_j = random.normal(keys[3], (n_j, n_ij))
    V_ij = random.normal(keys[4], (n_i + n_j, n_ij))
    U_ij = random.normal(keys[5], (n_i + n_j, n_ij))

    return (
        F_ij,
        F_ji,
        n_i,
        n_j,
        n_ij,
        S_i,
        S_j,
        S_ij,
        S_ji,
        Y_i,
        Y_j,
        U_i,
        U_j,
        V_ij,
        U_ij,
    )


@jit
def sylvester(
    A: jnp.ndarray,
    B: jnp.ndarray,
    C: jnp.ndarray,
) -> jnp.ndarray:
    """
    Solve AX + XB = C for X via Kronecker-product formulation:
    vec(X) = (I ⊗ A + B^T ⊗ I)^{-1} vec(C)
    """
    n, m = A.shape[0], B.shape[0]
    I_n = jnp.eye(n)
    I_m = jnp.eye(m)
    K = jnp.kron(I_m, A) + jnp.kron(B.T, I_n)
    vecC = C.reshape(n * m)
    vecX = jnp.linalg.solve(K, vecC)
    return vecX.reshape((n, m))


def polar_p(
    X: jnp.ndarray,
) -> jnp.ndarray:
    """Polar decomposition via SVD."""
    U, _, Vt = jnp.linalg.svd(X)
    return U @ Vt


@jit
def edge_body(
    state,
    cfg,
):
    """
    One iteration of ADMM for a single edge.
    """

    # Unpack the state
    (
        F_ij,
        F_ji,
        n_i,
        n_j,
        n_ij,
        S_i,
        S_j,
        S_ij,
        S_ji,
        Y_i,
        Y_j,
        U_i,
        U_j,
        V_ij,
        U_ij,
    ) = state
    alpha = cfg.algorithm.alpha
    lambda_ = cfg.algorithm.lambda_

    # Update F_ji via Sylvester
    A_j = jnp.eye(F_ij.shape[0]) - F_ij @ F_ij.T
    B_j = F_ij @ S_i @ F_ij.T + alpha * jnp.eye(F_ij.shape[0])
    C_j = (
        3 * F_ij @ S_ij
        + (alpha / 2) * (Y_j + V_ij[n_i:, :] - U_j - U_ij[n_i:, :]).T
    )
    F_ji = sylvester(
        jnp.linalg.inv(B_j) @ A_j,
        jnp.linalg.inv(S_j),
        jnp.linalg.inv(B_j) @ C_j @ jnp.linalg.inv(S_j),
    )
    # Update F_ij via Sylvester
    A_i = jnp.eye(n_ij) - F_ji @ F_ji.T
    B_i = F_ji @ S_j @ F_ji.T + alpha * jnp.eye(n_ij)
    C_i = (
        3 * F_ji @ S_ji
        + (alpha / 2) * (Y_i + V_ij[:n_i, :] - U_i - U_ij[:n_i, :]).T
    )
    F_ij = sylvester(
        jnp.linalg.inv(B_i) @ A_i,
        jnp.linalg.inv(S_i),
        jnp.linalg.inv(B_i) @ C_i @ jnp.linalg.inv(S_i),
    )

    # Store old values for convergence check
    # Y_i_prev = Y_i.copy()
    # Y_j_prev = Y_j.copy()
    # V_ij_prev = V_ij.copy()

    # Update dual variables
    Y_i = polar_p(F_ij.T + U_i)
    Y_j = polar_p(F_ji.T + U_j)
    V_ij = block_st(jnp.vstack((F_ij.T, F_ji.T)) + U_ij, beta=lambda_ / alpha)

    # Dual residuals
    # D_i = alpha * (Y_i - Y_i_prev)
    # D_j = alpha * (Y_j - Y_j_prev)
    # D_ij = alpha * (V_ij - V_ij_prev)

    # Primal residuals
    R_i = F_ij.T - Y_i
    R_j = F_ji.T - Y_j
    R_ij = jnp.vstack((F_ij.T, F_ji.T)) - V_ij

    # Update auxiliary variables
    U_i = U_i + R_i
    U_j = U_j + R_j
    U_ij = U_ij + R_ij

    new_state = (
        F_ij,
        F_ji,
        S_i,
        S_j,
        S_ij,
        S_ji,
        Y_i,
        Y_j,
        U_i,
        U_j,
        V_ij,
        U_ij,
    )
    return edge_body(new_state, cfg)


def make_edge_solver(cfg, n_iter):
    @jit
    def body(carry, _):
        new_carry, y = edge_body(carry)
        return new_carry, y

    @jit
    def run(state):
        final_state, _ = lax.scan(body, state, None, length=n_iter)
        return final_state

    return run


# @jit
# def update_edge(
#     state,
#     cfg,
#     n_iter: int,
# ):
#     """Return a jitted updater for a single edge with fixed n_iter."""
#     final_state = lax.scan(edge_body, (state, cfg), None, length=n_iter)
#     return final_state[0], final_state[1]


def sheaf_learn(
    cfg,
    net: Network,
) -> None:
    """Learning the sheaf restriction maps using JAX-jitted per-edge solver."""
    edges = net.graph.get_edgelist()
    n_iter = cfg.algorithm.n_iter

    # RNG
    seed = cfg.algorithm.seed
    base_key = random.PRNGKey(seed)
    keys = random.split(base_key, len(edges))
    for idx, edge in enumerate(edges):
        key = keys[idx]
        # Fetch current state
        init_state = init_edge_state(net=net, edge=edge, key=key)
        # run JIT solver
        solve_edge = make_edge_solver(cfg, n_iter)
        final_state = solve_edge(init_state)
        # back to NumPy
        F_ij_np = np.array(final_state[0])
        F_ji_np = np.array(final_state[1])
        net.update_restriction_maps(edge, F_ij_np, F_ji_np)
    return None


@hydra.main(
    config_path='../.conf/hydra/test',
    config_name='sheaf_learn',
    version_base='1.3',
)
def main(cfg) -> None:
    print('Starting the data generation...')
    net = generate_data(cfg)
    print('Data generation completed.')
    print('Starting the sheaf learning (JAX)...')
    sheaf_learn(cfg, net)
    print('Sheaf learning completed.')
    return None


if __name__ == '__main__':
    main()
