import sys
from pathlib import Path
import os

sys.path.append(str(Path(sys.path[0]).parent))

from src.sheaf import Network
from collections.abc import Callable
import polars as pl
import pickle


def memoize(func: Callable) -> Callable:
    """Decorator for file-based caching keyed by seed, n_agents, and p_edges."""

    def wrapper(
        cfg,
        net: Network,
    ) -> Network:
        cache_dir = os.path.join(
            str(Path(sys.path[0]).parent), '.cache', f'{cfg.network.path}'
        )

        os.makedirs(cache_dir, exist_ok=True)
        seed = getattr(cfg, 'seed', None)
        n_agents = cfg.network.n_agents
        p_edges = int(cfg.network.p_edges * 100)
        iters = cfg.algorithm.n_iter
        cache_file = os.path.join(
            cache_dir,
            f'sheaf_seed{seed}_agents{n_agents}_pedges{p_edges}_iters{iters}.pkl',
        )
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                cached_net = pickle.load(f)
            print(f'Loaded cached sheaf from {cache_file}')
            return cached_net

        result = func(cfg, net)
        with open(cache_file, 'wb') as f:
            pickle.dump(result, f)
        print(f'Saved sheaf structure in {cache_file}.')
        return result

    return wrapper


def save(func: Callable) -> Callable:
    """Decorator for file-based saving keyed by seed, beta, and lambda_."""

    def wrapper(
        cfg,
        net: Network,
        beta: float,
        lambda_: float = None,
    ) -> Network:
        cache_dir = os.path.join(
            str(Path(sys.path[0]).parent), '.cache', f'{cfg.network.path}'
        )

        os.makedirs(cache_dir, exist_ok=True)
        seed = getattr(cfg, 'seed', None)
        iters = cfg.algorithm.n_iter
        cache_file = os.path.join(
            cache_dir,
            f'sheaf_seed{seed}beta{beta}_lambda{lambda_}_iters{iters}.pkl',
        )

        result = func(cfg, net, beta, lambda_)
        with open(cache_file, 'wb') as f:
            pickle.dump(result, f)
        print(f'Saved sheaf structure in {cache_file}.')
        return result

    return wrapper


def save_metrics(
    cfg,
    metrics_type: str,
    metrics: dict,
    dict_type: str,
    res_path: Path,
) -> None:
    """Saves the metrics dictionary to a pickle file."""

    if dict_type == 'local_pca':
        metrics['explained_variance'] = cfg.coder.explained_variance
        metrics['seed'] = cfg.seed
        metrics['dict_type'] = dict_type
        metrics['simulation'] = cfg.simulation
    elif dict_type == 'learnable':
        metrics['seed'] = cfg.seed
        metrics['lambda'] = cfg.coder.regularizer
        metrics['dict_type'] = dict_type
        metrics['simulation'] = cfg.simulation
        metrics['augmented_multiplier_dict'] = (
            cfg.coder.augmented_multiplier_dict
        )
        metrics['augmented_multiplier_sparse'] = (
            cfg.coder.augmented_multiplier_sparse
        )
    else:
        raise ValueError(f"Unknown dictionary type '{dict_type}'.")
    filename = (
        (
            f'{metrics_type}_metrics_{cfg.simulation}_'
            + f'{cfg.coder.regularizer}_'
            + f'{cfg.coder.dict_type}_'
            + f'{cfg.coder.augmented_multiplier_dict}_'
            + f'{cfg.coder.augmented_multiplier_sparse}_'
            + f'sampling_strategy_{cfg.coder.subsampling_strategy}_'
            + f'{cfg.seed}.parquet'
        )
        if dict_type == 'learnable'
        else (
            f'{metrics_type}_metrics_{cfg.simulation}_'
            + f'{cfg.coder.explained_variance}_'
            + f'{cfg.coder.dict_type}_'
            + f'{cfg.seed}.parquet'
        )
    )
    df = pl.DataFrame(metrics)
    df.write_parquet(res_path / filename)
