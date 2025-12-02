import os
import sys
import pickle
import polars as pl
from pathlib import Path
from collections.abc import Callable

sys.path.append(str(Path(sys.path[0]).parent))

from src.sheaf import Network


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

    if dict_type is None:
        metrics['seed'] = cfg.seed
        metrics['lambda'] = None
        metrics['sparsity'] = None
        metrics['gamma'] = None
        metrics['dict_type'] = dict_type
        metrics['simulation'] = cfg.simulation
        metrics['augmented_multiplier_dict'] = None
        metrics['augmented_multiplier_sparse'] = None
        filename = (
            f'{metrics_type}_metrics_{cfg.simulation}_'
            + f'no_dict_{cfg.seed}.parquet'
        )
    elif dict_type == 'local_pca':
        metrics['explained_variance'] = cfg.coder.explained_variance
        metrics['seed'] = cfg.seed
        metrics['dict_type'] = dict_type
        metrics['simulation'] = cfg.simulation
        filename = (
            f'{metrics_type}_metrics_{cfg.simulation}_'
            + f'{cfg.coder.explained_variance}_'
            + f'{cfg.coder.dict_type}_'
            + f'{cfg.seed}.parquet'
        )
    elif dict_type == 'learnable':
        metrics['seed'] = cfg.seed
        metrics['lambda'] = cfg.coder.sparse_regularizer
        metrics['sparsity'] = cfg.coder.sparsity
        metrics['gamma'] = cfg.coder.dict_regularizer
        metrics['dict_type'] = dict_type
        metrics['simulation'] = cfg.simulation
        metrics['augmented_multiplier_dict'] = (
            cfg.coder.augmented_multiplier_dict
        )
        metrics['augmented_multiplier_sparse'] = (
            cfg.coder.augmented_multiplier_sparse
        )
        filename = (
            f'{metrics_type}_metrics_{cfg.simulation}_'
            + f'sparsity_{cfg.coder.sparsity}_'
            + f'{cfg.coder.sparse_regularizer}_'
            + f'{cfg.coder.dict_regularizer}_'
            + f'{cfg.coder.dict_type}_'
            + f'sampling_strategy_{cfg.coder.subsampling_strategy}_'
            + f'{cfg.seed}.parquet'
        )
    else:
        raise ValueError(f"Unknown dictionary type '{dict_type}'.")

    df = pl.DataFrame(metrics)
    df.write_parquet(res_path / filename)


def update_metrics(res_path, n_proto, current_accuracy, current_loss):
    """
    Loads accuracy and loss lists from a pickle file, updates them,
    and writes the updated lists back to the file.

    Parameters:
        res_path (str): Path to the pickle directory.
        current_accuracy (float): Accuracy for the current run.
        current_loss (float): Loss for the current run.
    """

    # pickle_path = res_path / 'network_performance.pkl'
    data_path = res_path / 'network_performance.parquet'

    dump = pl.read_parquet(data_path) if data_path.exists() else pl.DataFrame()

    # If file exists, load existing lists; otherwise initialize new ones
    # if os.path.exists(pickle_path):
    #     with open(pickle_path, 'rb') as f:
    #         data = pickle.load(f)

    #     accuracies = data.get('accuracies', [])
    #     losses = data.get('losses', [])
    # else:
    # accuracies = []
    # losses = []

    # # Update lists
    # accuracies.append(current_accuracy)
    # losses.append(current_loss)

    dump = dump.vstack(
        pl.DataFrame(
            {
                'n_proto': n_proto,
                'accuracies': [current_accuracy.tolist()],
                'edges_loss': [current_loss.tolist()],
            }
        )
    )

    dump.write_parquet(data_path)

    # Save back to pickle
    # with open(pickle_path, 'wb') as f:
    # pickle.dump(
    #     {
    #         'accuracies': accuracies,
    #         'losses': losses,
    #         'n_proto': n_proto,
    #     },
    #     f,
    # )

    print('Metrics updated successfully.')
    print(dump.filter(pl.col('n_proto') == n_proto))
    # print(f'Accuracies: {accuracies}')
    # print(f'Losses: {losses}')


def save_graphs(
    path,
    graph,
    graph_nodict,
):
    path = path / 'graphs.pkl'

    if not path.exists():
        data = []
    else:
        with open(path, 'rb') as f:
            data = pickle.load(f)

    data += [graph, graph_nodict]

    with open(path, 'wb') as f:
        pickle.dump(data, f)

    return None
