"""
This python module handles the network formation using numpy arrays.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(sys.path[0]).parent))

from src import Network, EdgeProblemProcrustes
from omegaconf import OmegaConf
from tqdm.auto import tqdm
from typing import Any
import polars as pl
import numpy as np
import hydra
import wandb


# def generate_data(cfg):
#     """Generate data using NumPy arrays."""
#     coder_params = OmegaConf.to_container(cfg.coder, resolve=True)
#     agents_info: dict[int, dict[str, Any]] = {}
#     models = OmegaConf.to_container(cfg.network.models, resolve=True)
#     for i, model in enumerate(models):
#         model_name = list(model.keys())[0]
#         agents_info[i] = {
#             'model': model_name,
#             'dataset': model[model_name][0],
#             'seed': model[model_name][1],
#         }

#     net = Network(
#         agents_info=agents_info,
#         coder_params=coder_params,
#     )
#     return net


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
    n_edges = cfg.algorithm.n_edges
    plotting_nclusters = cfg.algorithm.plotting_nclusters
    params = OmegaConf.to_container(cfg.algorithm, resolve=True)

    wandb_config = OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )

    # evs = [None, 0.9, 0.8, 0.7, 0.6, 0.5]
    # for ev in evs:
    #     print('Starting sheaf alignment...', end='\t')

    #     if ev is None:
    #         name = f'seed_{cfg.seed}_Procrustes_NoSparse'
    #         run = wandb.init(
    #             project=cfg.wandb.project,
    #             name=name,
    #             id=name,
    #             config=wandb_config,
    #         )
    #         # ================================================================
    #         #                         Sheaf Generation
    #         # ================================================================
    #         print('Starting sheaf generation...', end='\t')
    #         params['explained_variance'] = ev
    #         coder_params['explained_variance'] = ev
    #         for i, model in enumerate(models):
    #             model_name = list(model.keys())[0]
    #             agents_info[i] = {
    #                 'model': model_name,
    #                 'dataset': model[model_name][0],
    #                 'seed': model[model_name][1],
    #             }
    #         net = Network(
    #             agents_info=agents_info,
    #             coder_params=coder_params,
    #             run=run,
    #         )
    #         print('[Passed]')

    #         # ================================================================
    #         #                        Fitting Procrustes
    #         # ================================================================
    #         for edge in net.edges.values():
    #             prob = EdgeProblem(
    #                 edge=edge,
    #                 params=params,
    #                 run=run,
    #             )

    #             prob.fit()
    #     else:
    #         name = f'seed_{cfg.seed}_ev_{ev}_Procrustes'
    #         run = wandb.init(
    #             project=cfg.wandb.project,
    #             name=name,
    #             id=name,
    #             config=wandb_config,
    #         )
    #         # ================================================================
    #         #                         Sheaf Generation
    #         # ================================================================
    #         print('Starting sheaf generation...', end='\t')
    #         params['explained_variance'] = ev
    #         coder_params['explained_variance'] = ev
    #         for i, model in enumerate(models):
    #             model_name = list(model.keys())[0]
    #             agents_info[i] = {
    #                 'model': model_name,
    #                 'dataset': model[model_name][0],
    #                 'seed': model[model_name][1],
    #             }
    #         net = Network(
    #             agents_info=agents_info,
    #             coder_params=coder_params,
    #             run=run,
    #         )
    #         print('[Passed]')

    #         # ================================================================
    #         #                 Fitting Procrustes with PCA
    #         # ================================================================
    #         for edge in tqdm(list(net.edges.values())):
    #             prob = EdgeProblemProcrustes(
    #                 edge=edge,
    #                 params=params,
    #                 run=run,
    #             )

    #             prob.fit()

    #     net.update_graph(n_edges=n_edges)
    #     print('[Passed]')

    #     # ================================================================
    #     #                        Evaluation
    #     # ================================================================
    #     print('Starting maps evaluation...', end='\t')
    #     net.eval()
    #     wandb.finish()
    #     print('[Passed]')

    #     net.sheaf_plot(n_clusters=plotting_nclusters)

    # # coder_params['dict_type'] = None

    regs = [
        1e3,
    ]
    for reg in regs:
        rcode = np.random.randint(0, 1000)
        coder_params['regularizer'] = reg
        name = (
            f'atoms_{coder_params["n_atoms"]}_'
            + f'reg_{coder_params["regularizer"]}_'
            # + f'momentum_{coder_params["momentum_D"]}_'
            # + f'stepS{coder_params["Sstep"]}_'
            # + f'stepD{coder_params["Dstep"]}_'
            # + f'Diter{coder_params["D_iters"]}_'
            + f'{rcode}'
        )
        run = wandb.init(
            project=cfg.wandb.project,
            name=name,
            id=name,
            config=wandb_config,
        )
        # ================================================================
        #                         Sheaf Generation
        # ================================================================
        print('Starting sheaf generation...', end='\t')
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

        metrics = net.return_metrics()
        metrics['seed'] = cfg.seed
        metrics['lambda'] = cfg.coder.regularizer
        metrics['dict_type'] = cfg.coder.dict_type
        metrics['simulation'] = cfg.simulation
        metrics['augmented_multiplier_dict'] = (
            cfg.coder.augmented_multiplier_dict
        )
        metrics['augmented_multiplier_sparse'] = (
            cfg.coder.augmented_multiplier_sparse
        )
        df = pl.DataFrame(metrics)
        df.write_parquet(
            RESULTS
            / (
                f'dict_metrics_{cfg.simulation}_{reg}_'
                + f'{cfg.coder.dict_type}_'
                + f'{cfg.coder.augmented_multiplier_dict}_'
                + f'{cfg.coder.augmented_multiplier_sparse}_'
                + f'{cfg.seed}.parquet'
            )
        )

        # ================================================================
        #                Fitting Procrustes with Global Dict
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
        net.sheaf_plot(n_clusters=plotting_nclusters)
        wandb.finish()
        print('[Passed]')
    return None


if __name__ == '__main__':
    main()
