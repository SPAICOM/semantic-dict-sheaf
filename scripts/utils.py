import sys
from pathlib import Path
import os

sys.path.append(str(Path(sys.path[0]).parent))

from src.sheaf import Network
from collections.abc import Callable
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
