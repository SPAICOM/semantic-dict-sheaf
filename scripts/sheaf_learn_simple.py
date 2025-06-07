"""
This python module handles the network formation using numpy arrays.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(sys.path[0]).parent))

from src import Network, EdgeProblemProcrustes, EdgeProblem
from omegaconf import OmegaConf
from tqdm.auto import tqdm
from typing import Any
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
    coder_params = OmegaConf.to_container(cfg.coder, resolve=True)
    agents_info: dict[int, dict[str, Any]] = {}
    models = OmegaConf.to_container(cfg.network.models, resolve=True)
    n_edges = cfg.algorithm.n_edges
    params = OmegaConf.to_container(cfg.algorithm, resolve=True)

    evs = [None, 0.9, 0.8, 0.7, 0.6, 0.5]
    wandb_config = OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )

    for ev in evs:
        # ================================================================
        #                         Sheaf Generation
        # ================================================================
        params['explained_variance'] = ev
        coder_params['explained_variance'] = ev
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
        )

        print('OOOOOOOOOOOOOOO', ev)
        # ================================================================
        #                        Fitting Procrustes
        # ================================================================
        if ev is None:
            name = f'seed_{cfg.seed}_Procrustes_NoSparse'
            run = wandb.init(
                project=cfg.wandb.project,
                name=name,
                id=name,
                config=wandb_config,
            )
            for edge in net.edges.values():
                prob = EdgeProblem(
                    edge=edge,
                    params=params,
                    run=run,
                )

                prob.fit()
            wandb.finish()

        # ================================================================
        #                 Fitting Procrustes with PCA
        # ================================================================
        else:
            name = f'seed_{cfg.seed}_ev_{ev}_Procrustes'
            run = wandb.init(
                project=cfg.wandb.project,
                name=name,
                id=name,
                config=wandb_config,
            )
            for edge in tqdm(list(net.edges.values())):
                prob = EdgeProblemProcrustes(
                    edge=edge,
                    params=params,
                    run=run,
                )

                prob.fit()
            wandb.finish()

        net.update_graph(n_edges=n_edges)

    return net


if __name__ == '__main__':
    main()
