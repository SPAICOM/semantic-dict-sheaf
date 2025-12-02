"""
This python module handles semantic network formation .
"""

import sys
from pathlib import Path

from utils import save_metrics

sys.path.append(str(Path(sys.path[0]).parent))

from typing import Any

import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from tqdm.auto import tqdm

import wandb
from src import EdgeProblemProcrustes, Network


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
    wandb_config = OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )

    rcode = np.random.randint(0, 1000)
    name = 'test' + f'{rcode}'
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
    #                        Evaluation
    # ================================================================
    print('Starting maps evaluation...', end='\t')
    net.gt_heatmaps()
    net.global_dict_corr_heatmap()
    wandb.finish()
    print('[Passed]')

    return None


if __name__ == '__main__':
    main()
