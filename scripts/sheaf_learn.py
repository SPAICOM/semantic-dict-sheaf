"""
This python module handles the network formation using numpy arrays.
"""

import sys
from pathlib import Path
from utils import save_metrics

sys.path.append(str(Path(sys.path[0]).parent))

from src import Network, EdgeProblemProcrustes
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from tqdm.auto import tqdm
from typing import Any
import numpy as np
import hydra
import wandb


@hydra.main(
    config_path='../.conf/hydra/network',
    config_name='toy_example',
    version_base='1.3',
)
def main(cfg) -> None:
    CURRENT: Path = Path('.')
    RESULTS: Path = CURRENT / 'results'
    RESULTS.mkdir(exist_ok=True)

    coder_params = OmegaConf.to_container(cfg.coder, resolve=True)
    agents_info: dict[int, dict[str, Any]] = {}
    models = OmegaConf.to_container(cfg.network.models, resolve=True)
    n_edges = cfg.alignment.n_edges
    params = OmegaConf.to_container(cfg.alignment, resolve=True)
    align_problem = cfg.alignment.problem
    wandb_config = OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )

    # is_multirun = hc.mode == RunMode.MULTIRUN
    hc = HydraConfig.get()
    rcode = np.random.randint(0, 1000)
    if coder_params['dict_type'] == 'local_pca':
        name = (
            f'dict_type_{coder_params["dict_type"]}_'
            + f'ev_{coder_params["explained_variance"]}_'
            f'job{hc.job.num}' + f'{rcode}'
        )
    else:
        name = (
            f'dict_type_{coder_params["dict_type"]}_'
            f'atoms_{coder_params["n_atoms"]}_'
            + f'reg_{coder_params["regularizer"]}_'
            + f'subsample_{coder_params["n_subsampling"]}_'
            + f'subsample_{coder_params["subsampling_strategy"]}_'
            # + f'initD_{coder_params["init_mode_dict"]}_'
            # + f'momentum_{coder_params["momentum_D"]}_'
            # + f'stepS{coder_params["Sstep"]}_'
            # + f'stepD{coder_params["Dstep"]}_'
            # + f'Diter{coder_params["D_iters"]}_'
            + f'maxiter{coder_params["max_iter"]}_'
            f'job{hc.job.num}' + f'{rcode}'
        )

    run = wandb.init(
        project=cfg.wandb.project,
        name=name,
        id=name,
        group=cfg.simulation,
        config=wandb_config,
    )
    # ================================================================
    #             Sheaf Generation + Dictionary Learning
    # ================================================================
    print('Starting sheaf generation and latents sparse coding...', end='\t')
    for i, model in enumerate(models):
        model_name = list(model.keys())[0]
        agents_info[i] = {
            'model': model_name,
            'dataset': model[model_name][0],
            'seed': model[model_name][1],
        }
    net = Network(
        agents_info=agents_info,
        coder_params=coder_params,
        run=run,
    )
    print('[Passed]')

    # ================================================================
    #                        Sheaf Alignment
    # ================================================================
    if align_problem == 'procrustes':
        # ================================================================
        #                    Procrustes on Edges
        # ================================================================
        for edge in tqdm(list(net.edges.values())):
            prob = EdgeProblemProcrustes(
                edge=edge,
                params=params,
                run=run,
            )

            prob.fit()

    net.update_graph(n_edges=n_edges)
    print('[Passed]')

    # ================================================================
    #                        Evaluation
    # ================================================================
    print('Starting maps evaluation...', end='\t')
    net.eval()
    net.sheaf_plot(
        n_clusters=cfg.visualization.nclusters,
        layout=cfg.visualization.layout,
    )
    if cfg.visualization.plot_restriction_maps:
        net.restriction_maps_heatmap(cfg.alignment.n_edges)
    if cfg.visualization.plot_pca_correlation:
        net.pca_correlation_heatmap(20, cfg.alignment.n_edges)
    wandb.finish()
    print('[Passed]')

    if cfg.save_results:
        # ================================================================
        #                   Save Sparse Coding Metrics
        # ================================================================
        metrics = net.return_dict_metrics()
        save_metrics(
            cfg=cfg,
            metrics_type='dict',
            metrics=metrics,
            dict_type=cfg.coder.dict_type,
            res_path=RESULTS,
        )

        # ================================================================
        #                   Save Alignment Metrics
        # ================================================================
        metrics = net.return_alignment_metrics()
        save_metrics(
            cfg=cfg,
            metrics_type='alignment',
            metrics=metrics,
            dict_type=cfg.coder.dict_type,
            res_path=RESULTS,
        )
    return None


if __name__ == '__main__':
    main()
