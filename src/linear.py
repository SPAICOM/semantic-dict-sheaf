""""""

import sys
from pathlib import Path

sys.path.append(str(Path(sys.path[0]).parent))

import torch
from torch.utils.data import TensorDataset, DataLoader
from pytorch_lightning import Trainer
from typing import Any
from igraph import Graph
from tqdm.auto import tqdm
import jax.numpy as jnp
from src.datamodules import DataModuleClassifier
from src.neural import Classifier


class Agent:
    """ """

    def __init__(
        self,
        id: int,
        model: str,
        dataset: str = 'cifar10',
        device: str = 'cpu',
    ):
        self.id: int = id
        self.model_name: str = model
        self.dataset: str = dataset
        self.device: str = device

        # Other variables
        self.restriction_maps: dict[int, torch.Tensor] = {}

        self.latent_initialization()

    def map_initialization(
        self,
        neighbour_id: int,
        edge_stalk_dim: int,
    ) -> None:
        """ """
        self.restriction_maps[neighbour_id] = torch.randn(
            (edge_stalk_dim, self.stalk_dim)
        ).to(self.device)
        return None

    def map_update(
        self,
        neighbour_id: int,
        map_value: torch.Tensor,
    ) -> None:
        """ """
        # TODO: Check if the map_value is a torch tensor, otherwise convert it
        # to a torch tensor.
        self.restriction_maps[neighbour_id] = map_value
        return None

    def latent_initialization(
        self,
        dataset: str = 'cifar10',
    ) -> None:
        """ """
        self.datamodule = DataModuleClassifier(
            dataset=dataset, rx_enc=self.model_name
        )
        self.datamodule.prepare_data()
        self.datamodule.setup()
        self.X_train = self.datamodule.train_data.input.T
        self.X_test = self.datamodule.test_data.input.T
        self.stalk_dim = self.X_train.shape[0]

        return None

    def load_model(
        self,
        model_path: str,
    ) -> None:
        """ """
        self.model = Classifier.load_from_checkpoint(model_path)
        self.model.eval()
        return None

    def test_model(
        self,
        data: torch.Tensor,
        trainer: Trainer,
    ) -> tuple[float, float]:
        """ """
        dl = DataLoader(
            TensorDataset(data.T, self.datamodule.test_data.labels),
            batch_size=self.datamodule.batch_size,
        )
        res = trainer.test(model=self.model, dataloaders=dl)
        return res[0]['test/acc_epoch'], res[0]['test/loss_epoch']


class Network:
    """ """

    def __init__(
        self,
        agents_info: dict[int, dict[str, Any]],
        device: str = 'cpu',
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.n_agents: int = len(agents_info)
        self.agents_info: dict[int, dict[str, Any]] = agents_info
        self.device: str = device

        # Initialize the Agents
        self.agents: dict[int, Agent] = {}
        for idx, info in self.agents_info.items():
            self.agents[idx] = Agent(
                id=idx,
                model=info['model'],
                dataset=info['dataset'],
                device=self.device,
            )

    def graph_initialization(
        self,
        edges: list[tuple[int, int]] = None,
        edges_stalks: dict[tuple[int, int], int] = None,
        p: float = 1.0,
    ) -> None:
        """ """
        if edges is None:
            self.graph = Graph.Erdos_Renyi(
                n=self.n_agents, p=p, directed=False, loops=False
            )
        else:
            self.graph = Graph()
            self.graph.add_edges(edges)

        for i, j in tqdm(self.graph.get_edgelist()):
            cap = min(self.agents[i].stalk_dim, self.agents[j].stalk_dim)

            if edges_stalks is not None:
                e_stalk = edges_stalks[(i, j)]
            else:
                e_stalk = torch.randint(low=1, high=cap, size=(1, 1)).item()

            assert e_stalk <= cap, (
                'The edge stalk must be smaller than the minimum agent stalk.'
            )

            self.agents[i].map_initialization(
                neighbour_id=j, edge_stalk_dim=e_stalk
            )
            self.agents[j].map_initialization(
                neighbour_id=i, edge_stalk_dim=e_stalk
            )

        return None

    def update_restriction_maps(
        self,
        edge: tuple[int, int],
        F_ij: torch.Tensor,
        F_ji: torch.Tensor,
    ) -> None:
        """ """
        i, j = edge
        self.agents[i].map_update(j, F_ij)
        self.agents[j].map_update(i, F_ji)

        return None

    def get_sample_covs(
        self,
        edge: tuple[int, int],
        out: str = 'np',
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """ """
        assert out in [
            'torch',
            'np',
            'jax',
        ], (
            "Output type must be either 'torch' (torch.Tensor), 'numpy' (np.ndarray), or 'jax' (jax.array)."
        )

        i, j = edge
        S_i = self.agents[i].X_train @ self.agents[i].X_train.T
        S_j = self.agents[j].X_train @ self.agents[j].X_train.T
        S_ij = self.agents[i].X_train @ self.agents[j].X_train.T
        S_ji = self.agents[j].X_train @ self.agents[i].X_train.T

        if out == 'np':
            S_i = S_i.detach().cpu().numpy()
            S_j = S_j.detach().cpu().numpy()
            S_ij = S_ij.detach().cpu().numpy()
            S_ji = S_ji.detach().cpu().numpy()
        elif out == 'jax':
            S_i = jnp.array(S_i.detach().cpu().numpy())
            S_j = jnp.array(S_j.detach().cpu().numpy())
            S_ij = jnp.array(S_ij.detach().cpu().numpy())
            S_ji = jnp.array(S_ji.detach().cpu().numpy())

        return S_i, S_j, S_ij, S_ji

    def data_initialization(self) -> None:
        """ """
        for i in range(self.n_agents):
            self.agents[i].latent_initialization(
                self.agents_info[i]['dataset']
            )
        return None

    def send_message(
        self,
        tx_id: int,
        rx_id: int,
        message: torch.Tensor,
    ) -> torch.Tensor:
        """ """
        return (
            self.agents[rx_id].restriction_maps[tx_id].T
            @ self.agents[tx_id].restriction_maps[rx_id]
            @ message
        )

    def _edge_loss(
        self,
        edge: tuple[int, int],
        lambda_: float,
    ) -> float:
        """Compute the edge loss between two agents."""
        i, j = edge
        F_ij = self.agents[i].restriction_maps[j]
        F_ji = self.agents[j].restriction_maps[i]
        X_i = self.agents[i].X_train
        X_j = self.agents[j].X_train
        # Compute the Frobenius norm of the difference between the two maps
        loss = (
            torch.norm(F_ij @ X_i - F_ji @ X_j, p='fro')
            + torch.norm(X_i - F_ij.T @ F_ji @ X_j, p='fro')
            + torch.norm(X_j - F_ji.T @ F_ij @ X_i, p='fro')
            + lambda_
            * torch.norm(torch.vstack([F_ij, F_ji]), p=2, dim=0).sum()
        )
        return loss

    def compute_edge_losses(
        self, lambda_: float
    ) -> dict[tuple[int, int], float]:
        """Compute the edge losses for all edges in the graph."""
        edge_losses: dict[tuple[int, int], float] = {}
        for i, j in self.graph.get_edgelist():
            edge_losses[(i, j)] = self._edge_loss(edge=(i, j), lambda_=lambda_)
        return edge_losses

    def update_graph(self, n_edges: int, lambda_: float) -> None:
        """Update the graph based on the edge losses."""
        assert (n_edges > 0) and (n_edges <= len(self.graph.get_edgelist())), (
            'n_edges must be a positive integer, smaller than the current number of edges in the graph.'
        )
        edge_losses = self.compute_edge_losses(lambda_=lambda_)
        to_remove = list(
            dict(sorted(edge_losses.items(), key=lambda item: item[1])).keys()
        )[n_edges:]
        eids = self.graph.get_eids(to_remove, directed=False, error=False)
        eids = [eid for eid in eids if eid >= 0]
        self.graph.delete_edges(eids)

    def test_agent_model(
        self,
        agent_id: int,
        dataset: str = 'cifar10',
        seed: int = 42,
    ) -> torch.Tensor:
        """ """
        trainer: Trainer = Trainer(
            inference_mode=True,
            enable_progress_bar=False,
            logger=False,
            accelerator=self.device,
        )
        losses: dict[int, float] = {}
        accuracy: dict[str, float] = {}
        neighbors: list[int] = self.graph.neighbors(agent_id)
        for rx_id in neighbors:
            # Send the message from the input agent to one of its neighbors
            rx_data = self.send_message(
                tx_id=agent_id,
                rx_id=rx_id,
                message=self.agents[agent_id].X_test,
            )

            self.agents[rx_id].load_model(
                model_path=f'models/classifiers/{dataset}/{self.agents[rx_id].model_name}/seed_{seed}.ckpt'
            )
            acc, loss = self.agents[rx_id].test_model(
                data=rx_data,
                trainer=trainer,
            )
            accuracy[
                f'Agent-{rx_id} ({self.agents[rx_id].model_name}) - Task Accuracy (Test)'
            ] = acc

            losses[
                f'Agent-{rx_id} ({self.agents[rx_id].model_name}) - MSE loss (Test)'
            ] = loss

        return torch.mean(torch.tensor(list(accuracy.values()))), torch.mean(
            torch.tensor(list(losses.values()))
        )

    def eval(
        self,
        dataset: str = 'cifar10',
        seed: int = 42,
        verbose: bool = False,
    ) -> None:
        """ """
        agents_metrics: dict[int, tuple[float, float]] = {}
        for i in range(self.n_agents):
            acc, loss = self.test_agent_model(
                agent_id=i,
                dataset=dataset,
                seed=seed,
            )
            if verbose:
                print(
                    f'Agent-{i} ({self.agents[i].model_name}) - Task Accuracy (Test): {acc}'
                )
                print(
                    f'Agent-{i} ({self.agents[i].model_name}) - MSE loss (Test): {loss}'
                )
            agents_metrics[i] = (acc, loss)
        return agents_metrics


def main():
    """The main loop."""
    print('Start performing sanity tests...')

    # Variables definition
    agents_info: dict[int, dict[str, Any]] = {
        0: {'model': 'vit_base_patch16_224', 'dataset': 'cifar10'},
        1: {'model': 'vit_base_patch16_224', 'dataset': 'cifar10'},
    }
    device: str = 'cpu'

    print('First Test...', end='\t')
    net = Network(agents_info=agents_info, device=device)
    net.graph_initialization()
    print(net.agents[0].stalk_dim, net.agents[1].stalk_dim)
    print('[Passed]')

    return None


if __name__ == '__main__':
    main()
