""""""

from pathlib import Path
import sys

sys.path.append(str(Path(sys.path[0]).parent))

from torch.utils.data import TensorDataset, DataLoader
from pytorch_lightning import Trainer
from scipy.linalg import dft
from tqdm.auto import tqdm
from igraph import Graph
import jax.numpy as jnp
from typing import Any
import numpy as np
import torch

from src.utils import random_stiefel, convert_output, convert_input, prewhiten
from src.datamodules import DataModuleClassifier
from src.coder import SparseCoder
from src.neural import Classifier


# def convert_output(fn):
#     @wraps(fn)
#     def wrapper(self, *args, out: str = 'torch', **kwargs):
#         result = fn(self, *args, **kwargs)

#         def convert(x: torch.Tensor):
#             if x is None:
#                 return x
#             if out == 'numpy':
#                 return x.detach().cpu().numpy()
#             elif out == 'jax':
#                 return jnp.array(x.detach().cpu().numpy())
#             elif out == 'torch':
#                 return x
#             else:
#                 raise ValueError(f"Unsupported out='{out}'")

#         if isinstance(result, tuple):
#             return tuple(convert(x) for x in result)
#         else:
#             return convert(result)

#     return wrapper


class Agent:
    """ """

    def __init__(
        self,
        id: int,
        model: str,
        seed: int,
        dataset: str = 'cifar10',
        device: str = 'cpu',
        testing: bool = False,
        **kwargs: Any,
    ):
        self.id: int = id
        self.model_name: str = model
        self.dataset: str = dataset
        self.device: str = device
        self.seed: int = seed
        self.S: torch.Tensor = None
        self.restriction_maps: dict[int, torch.Tensor] = {}
        # self.sparse_representations: dict[int, torch.Tensor] = {}

        if testing:
            self.X_train = kwargs['X_train']
            self.X_test = None
            self.stalk_dim = self.X_train.shape[0]
        else:
            self.latent_initialization()

    def map_initialization(
        self,
        neighbour_id: int,
        edge_stalk_dim: int,
        orthogonal: bool = False,
    ) -> None:
        """ """
        if orthogonal:
            self.restriction_maps[neighbour_id] = random_stiefel(
                edge_stalk_dim, self.stalk_dim
            ).to(self.device)
        else:
            self.restriction_maps[neighbour_id] = torch.randn(
                edge_stalk_dim,
                self.stalk_dim,
                device=self.device,
            )

        return None

    def sparse_initialization(
        self,
        coder_params: dict[Any] = {},
    ) -> None:
        """ """
        X, _ = self.get_latent(prewhite=True, out='numpy')
        coder = SparseCoder(
            X=X,
            params=coder_params,
        )
        self.D, self.S = coder.fit(out='torch')
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
        self.n_examples = self.X_train.shape[1]
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

    @convert_output
    def get_latent(
        self,
        prewhite: bool = False,
        test: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if test:
            X_tr, _, _, X_te = (
                prewhiten(self.X_train, self.X_test)
                if prewhite
                else (self.X_train, None, None, self.X_test),
            )
        else:
            X_tr, _, _ = (
                prewhiten(self.X_train)
                if prewhite
                else (self.X_train, None, None)
            )
            X_te = None
        return X_tr, X_te


class Edge:
    """"""

    def __init__(
        self,
        head: Agent,
        tail: Agent,
        stalk_dim: int,
        mask: bool = False,
        device: str = 'cpu',
    ):
        self.head = head
        self.tail = tail
        self.id = (self.head.id, self.tail.id)
        self.stalk_dim = stalk_dim
        self.device = device
        self.mask: torch.Tensor = (
            torch.eye(self.stalk_dim, device=self.device) if mask else None
        )
        self.loss: float = torch.inf

    @convert_output
    def get_restriction_maps(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """"""
        F_i = self.head.restriction_maps[self.id[1]]
        F_j = self.tail.restriction_maps[self.id[0]]
        return F_i, F_j

    @convert_output
    def get_sparse_representations(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """"""
        S_i = self.head.S
        S_j = self.tail.S
        return S_i, S_j

    @convert_output
    def get_dictionaries(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.head.D, self.tail.D

    @convert_output
    def get_latents(
        self,
        set: str = 'train',
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert set in ('train', 'test'), (
            '"set" argument must be either "train" or "test".'
        )
        if set == 'train':
            X_i = self.head.X_train
            X_j = self.head.X_train
        else:
            X_i = self.head.X_test
            X_j = self.head.X_test
        return X_i, X_j

    @convert_output
    def get_latent_covariances(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """ """
        cov_i = self.head.X_train @ self.head.X_train.T
        cov_j = self.tail.X_train @ self.tail.X_train.T
        cov_ij = self.head.X_train @ self.tail.X_train.T
        cov_ji = self.tail.X_train @ self.head.X_train.T
        return cov_i, cov_j, cov_ij, cov_ji

    @convert_output
    def get_mask(
        self,
    ) -> torch.Tensor:
        return self.mask

    def update_restriction_maps(
        self,
        F_i: torch.Tensor | np.ndarray | jnp.ndarray,
        F_j: torch.Tensor | np.ndarray | jnp.ndarray,
    ) -> None:
        """
        Update the restriction maps for this edge.
        F_i corresponds to head.restriction_maps[tail_id],
        F_j corresponds to tail.restriction_maps[head_id].
        """
        F_i = convert_input(F_i, self.device)
        F_j = convert_input(F_j, self.device)

        self.head.restriction_maps[self.id[1]] = F_i
        self.tail.restriction_maps[self.id[0]] = F_j
        return None

    def update_sparse_representations(
        self,
        S_i: torch.Tensor | np.ndarray | jnp.ndarray,
        S_j: torch.Tensor | np.ndarray | jnp.ndarray,
    ) -> None:
        """
        Update the sparse representations for the agents corresponding to this edge.
        S_i corresponds to head.sparse_representations[tail_id],
        S_j corresponds to tail.sparse_representations[head_id].
        """
        S_i = convert_input(S_i, self.device)
        S_j = convert_input(S_j, self.device)

        self.head.S = S_i
        self.tail.S = S_j
        return None

    def update_mask(
        self,
        Z_ij: torch.Tensor | np.ndarray | jnp.ndarray,
    ) -> None:
        """Update the edge stalk mask for this edge."""
        self.mask = convert_input(Z_ij, self.device)
        return None

    def update_loss(
        self,
        loss: float,
    ) -> None:
        self.loss = loss
        return None


class Network:
    """ """

    def __init__(
        self,
        agents_info: dict[int, dict[str, Any]],
        edges_info: dict[tuple[int, int], int] = None,
        coder_params: dict[Any] = {},
        semantic_compression: bool = False,
        dictionary: torch.Tensor = None,
        mask_edges: bool = False,
        device: str = 'cpu',
    ):
        self.edges: dict[tuple[int, int], Edge] = {}
        self.agents: dict[int, Agent] = {}
        self.is_connection_graph: bool = False
        self.dictionary = dictionary
        self.device: str = device
        self.global_dim: int = 0
        self.n_agents: int = 0
        self.n_edges: int = 0

        # Init Agents
        for idx, info in agents_info.items():
            self.agents[idx] = Agent(
                id=idx,
                model=info['model'],
                dataset=info['dataset'],
                seed=info['seed'],
                device=self.device,
                testing=info.get('testing', False),
                **info.get('kwargs', {}),
            )
        self.n_agents = len(self.agents)

        # Init Edges
        if edges_info is None:
            self.graph = Graph.Erdos_Renyi(
                n=self.n_agents,
                p=1.0,
                directed=False,
                loops=False,
            )
        else:
            self.graph = Graph()
            self.graph.add_vertices(self.n_agents)
            self.graph.add_edges(list(edges_info.keys()))

        for i, j in tqdm(self.graph.get_edgelist()):
            cap = min(self.agents[i].stalk_dim, self.agents[j].stalk_dim)
            if edges_info is not None:
                e_stalk = edges_info[(i, j)]
            else:
                e_stalk = (
                    torch.randint(low=1, high=cap, size=(1, 1)).item()
                    if semantic_compression
                    else cap
                )
            del cap
            self.agents[i].map_initialization(
                neighbour_id=j, edge_stalk_dim=e_stalk
            )
            self.agents[j].map_initialization(
                neighbour_id=i, edge_stalk_dim=e_stalk
            )
            self.agents[i].sparse_initialization(coder_params=coder_params)
            self.agents[j].sparse_initialization(coder_params=coder_params)
            self.edges[(i, j)] = Edge(
                head=self.agents[i],
                tail=self.agents[j],
                stalk_dim=e_stalk,
                mask=mask_edges,
                device=self.device,
            )
        self.n_edges = len(self.edges)
        self._is_connection_graph()
        del agents_info, edges_info, coder_params

    def _is_connection_graph(
        self,
    ) -> None:
        """
        Check if the network sheaf is a connection graph, i.e.,
        if node and edge stalks all have the same dimension.
        """
        n = self.agents[0].restriction_maps[1].shape[1]
        node_dims = np.array(
            [agent.stalk_dim for agent in self.agents.values()]
        )
        edge_dims = np.array([edge.stalk_dim for edge in self.edges.values()])
        self.is_connection_graph = np.all(node_dims == n) & np.all(
            edge_dims == n
        )
        self.global_dim: int = n if self.is_connection_graph else None
        del edge_dims, node_dims, n
        return None

    def update_graph(
        self,
        n_edges: int,
    ) -> None:
        """Update the graph based on the edge losses."""
        assert (n_edges > 0) and (n_edges <= len(self.graph.get_edgelist())), (
            'n_edges must be a positive integer, smaller than the current number of edges in the graph.'
        )

        edge_losses = {edge: edge.loss for edge in self.edges}
        to_remove = list(
            dict(sorted(edge_losses.items(), key=lambda item: item[1])).keys()
        )[n_edges:]
        eids = self.graph.get_eids(to_remove, directed=False, error=False)
        eids = [eid for eid in eids if eid >= 0]
        self.graph.delete_edges(eids)

    #######################################################################################

    # def update_restriction_maps(
    #     self,
    #     edge: tuple[int, int],
    #     F_j: torch.Tensor,
    #     F_i: torch.Tensor = None,
    #     Z_ij: torch.Tensor = None,
    # ) -> None:
    #     """ """
    #     i, j = edge
    #     if F_i is None:
    #         F_i = torch.eye(self.agents[i].stalk_dim, device=self.device)
    #     self.agents[i].map_update(j, F_j)
    #     self.agents[j].map_update(i, F_i)
    #     self.edges[edge] = Z_ij
    #     return None

    # def send_message(
    #     self,
    #     tx_id: int,
    #     rx_id: int,
    #     message: torch.Tensor,
    # ) -> torch.Tensor:
    #     """ """
    #     if self.edge_masks[(tx_id, rx_id)] is None:
    #         return (
    #             self.agents[rx_id].restriction_maps[tx_id].T
    #             @ self.agents[tx_id].restriction_maps[rx_id]
    #             @ message
    #         )
    #     return (
    #         self.agents[rx_id].restriction_maps[tx_id].T
    #         @ self.edge_masks[(tx_id, rx_id)]
    #         @ self.agents[tx_id].restriction_maps[rx_id]
    #         @ message
    #     )

    # def edge_loss(
    #     self,
    #     edge: tuple[int, int],
    #     S_t: torch.Tensor = None,
    #     beta: float = None,
    #     penalized: bool = False,
    #     lambda_: float = None,
    #     F_ij: torch.Tensor = None,
    #     F_ji: torch.Tensor = None,
    #     S_i: torch.Tensor = None,
    #     S_j: torch.Tensor = None,
    #     D: torch.Tensor = None,
    #     gamma1: float = None,
    #     gamma2: float = None,
    # ) -> tuple[float, float, float]:
    #     """Compute the edge loss between two agents."""
    #     i, j = edge
    #     F_ij = self.agents[i].restriction_maps[j] if F_ij is None else F_ij
    #     F_ji = self.agents[j].restriction_maps[i] if F_ji is None else F_ji
    #     S_i = self.agents[i].sparse_representations[j] if S_i is None else S_i
    #     S_j = self.agents[j].sparse_representations[i] if S_j is None else S_i
    #     X_i = self.agents[i].X_train
    #     X_j = self.agents[j].X_train
    #     semantic_alignment = (
    #         torch.norm(F_ij @ X_i - F_ji @ X_j, p='fro')
    #         if S_t is None
    #         else torch.norm(S_t @ (F_ij @ X_i - F_ji @ X_j), p='fro')
    #     )
    #     communication_loss = (
    #         torch.norm(X_i - F_ij.T @ F_ji @ X_j, p='fro')
    #         + torch.norm(X_j - F_ji.T @ F_ij @ X_i, p='fro')
    #         if S_t is None
    #         else torch.norm(X_i - F_ij.T @ S_t @ F_ji @ X_j, p='fro')
    #         + torch.norm(X_j - F_ji.T @ S_t @ F_ij @ X_i, p='fro')
    #     )
    #     loss = (
    #         semantic_alignment
    #         if self.is_connection_graph
    #         else (1 - beta) * semantic_alignment + beta * communication_loss
    #     )

    #     assert (penalized == True) + (S_t is not None) != 1, (
    #         'penalized must be True if S_t is not None, and vice versa.'
    #     )
    #     if penalized:
    #         loss += lambda_ * torch.norm(torch.diag(S_t), p=1)

    #     if S_i is not None:
    #         loss += (
    #             torch.norm(X_i - D @ S_i, p=2)
    #             + torch.norm(X_j - D @ S_j, p=2)
    #             + gamma1 * torch.linalg.norm(S_j, ord=2, dim=1).sum()
    #             + gamma2 * torch.linalg.norm(S_i, ord=2, dim=1).sum()
    #         )
    #     return loss, semantic_alignment, communication_loss

    # def edge_augmented_lagrangian(
    #     self,
    #     edge: tuple[int, int],
    #     S_t: np.ndarray,
    #     alpha: float,
    #     U_i: np.ndarray,
    #     U_j: np.ndarray,
    #     Y_i: np.ndarray,
    #     Y_j: np.ndarray,
    #     beta: float = None,
    #     penalized: bool = None,
    #     lambda_: float = None,
    #     F_ij: torch.Tensor = None,
    #     F_ji: torch.Tensor = None,
    # ) -> float:
    #     U_i = torch.from_numpy(U_i.astype(np.float32))
    #     U_j = torch.from_numpy(U_j.astype(np.float32))
    #     Y_i = torch.from_numpy(Y_i.astype(np.float32))
    #     Y_j = torch.from_numpy(Y_j.astype(np.float32))
    #     S_t = (
    #         torch.from_numpy(S_t.astype(np.float32))
    #         if S_t is not None
    #         else None
    #     )
    #     F_ij = (
    #         torch.from_numpy(F_ij.astype(np.float32))
    #         if F_ij is not None
    #         else F_ij
    #     )
    #     F_ji = (
    #         torch.from_numpy(F_ji.astype(np.float32))
    #         if F_ji is not None
    #         else F_ji
    #     )
    #     loss, sem, comm = self.edge_loss(
    #         edge=edge,
    #         S_t=S_t,
    #         beta=beta,
    #         penalized=penalized,
    #         lambda_=lambda_,
    #         F_ij=F_ij,
    #         F_ji=F_ji,
    #     )
    #     dual_loss = (alpha / 2) * (
    #         torch.norm(F_ji.T - Y_j + U_j, p='fro') ** 2
    #         + torch.norm(F_ij.T - Y_i + U_i, p='fro') ** 2
    #     )
    #     return loss, sem, comm, dual_loss

    # def edge_loss_grad(
    #     self,
    #     edge: tuple[int, int],
    #     S_t: np.ndarray,
    #     beta: float,
    #     autodiff: bool = True,
    # ) -> np.ndarray:
    #     """Compute the edge loss gradient."""
    #     if autodiff:
    #         S_t = torch.tensor(
    #             S_t,
    #             dtype=torch.float32,
    #             device=self.device,
    #             requires_grad=True,
    #         )
    #         loss, _, _ = self.edge_loss(
    #             edge=edge,
    #             S_t=S_t,
    #             beta=beta,
    #             penalized=False,
    #         )
    #         loss.backward()
    #         # return S_t.grad.detach().cpu().numpy()
    #         return S_t.grad.numpy()
    #     else:
    #         return None

    # def compute_edge_losses(
    #     self,
    #     dictionary: torch.Tensor = None,
    #     beta: float = None,
    #     lambda_: float = None,
    #     gamma1: float = None,
    #     gamma2: float = None,
    # ) -> dict[tuple[int, int], float]:
    #     """Compute the edge losses for all edges in the graph."""
    #     edge_losses: dict[tuple[int, int], float] = {}
    #     penalized = False if lambda_ is None else True
    #     for i, j in self.graph.get_edgelist():
    #         edge_losses[(i, j)], _, _ = self.edge_loss(
    #             edge=(i, j),
    #             S_t=self.edge_masks[(i, j)],
    #             beta=beta,
    #             penalized=penalized,
    #             lambda_=lambda_,
    #             gamma1=gamma1,
    #             gamma2=gamma2,
    #             D=dictionary,
    #         )
    #     return edge_losses

    # def update_graph(
    #     self,
    #     n_edges: int,
    #     dictionary: torch.Tensor = None,
    #     beta: float = None,
    #     lambda_: float = None,
    #     gamma1: float = None,
    #     gamma2: float = None,
    # ) -> None:
    #     """Update the graph based on the edge losses."""
    #     assert (n_edges > 0) and (n_edges <= len(self.graph.get_edgelist())), (
    #         'n_edges must be a positive integer, smaller than the current number of edges in the graph.'
    #     )

    #     edge_losses = self.compute_edge_losses(
    #         beta=beta,
    #         lambda_=lambda_,
    #         gamma1=gamma1,
    #         gamma2=gamma2,
    #         dictionary=dictionary,
    #     )
    #     to_remove = list(
    #         dict(sorted(edge_losses.items(), key=lambda item: item[1])).keys()
    #     )[n_edges:]
    #     eids = self.graph.get_eids(to_remove, directed=False, error=False)
    #     eids = [eid for eid in eids if eid >= 0]
    #     self.graph.delete_edges(eids)

    # def test_agent_model(
    #     self,
    #     agent_id: int,
    #     dataset: str = 'cifar10',
    #     seed: int = 42,
    # ) -> torch.Tensor:
    #     """ """
    #     trainer: Trainer = Trainer(
    #         inference_mode=True,
    #         enable_progress_bar=False,
    #         logger=False,
    #         accelerator=self.device,
    #     )
    #     losses: dict[int, float] = {}
    #     accuracy: dict[str, float] = {}
    #     neighbors: list[int] = self.graph.neighbors(agent_id)
    #     for rx_id in neighbors:
    #         # Send the message from the input agent to one of its neighbors
    #         rx_data = self.send_message(
    #             tx_id=agent_id,
    #             rx_id=rx_id,
    #             message=self.agents[agent_id].X_test,
    #         )

    #         self.agents[rx_id].load_model(
    #             model_path=f'models/classifiers/{dataset}/{self.agents[rx_id].model_name}/seed_{self.agents[rx_id].seed}.ckpt'
    #         )
    #         acc, loss = self.agents[rx_id].test_model(
    #             data=rx_data,
    #             trainer=trainer,
    #         )
    #         accuracy[
    #             f'Agent-{rx_id} ({self.agents[rx_id].model_name}) - Task Accuracy (Test)'
    #         ] = acc

    #         losses[
    #             f'Agent-{rx_id} ({self.agents[rx_id].model_name}) - MSE loss (Test)'
    #         ] = loss

    #     return torch.mean(torch.tensor(list(accuracy.values()))), torch.mean(
    #         torch.tensor(list(losses.values()))
    #     )

    # def eval(
    #     self,
    #     dataset: str = 'cifar10',
    #     seed: int = 42,
    #     verbose: bool = False,
    # ) -> None:
    #     """ """
    #     agents_metrics: dict[int, tuple[float, float]] = {}
    #     for i in range(self.n_agents):
    #         acc, loss = self.test_agent_model(
    #             agent_id=i,
    #             dataset=dataset,
    #             seed=seed,
    #         )
    #         if verbose:
    #             print(
    #                 f'Agent-{i} ({self.agents[i].model_name}) - Task Accuracy (Test): {acc}'
    #             )
    #             print(
    #                 f'Agent-{i} ({self.agents[i].model_name}) - MSE loss (Test): {loss}'
    #             )
    #         agents_metrics[i] = (acc, loss)
    #     return agents_metrics


def set_global_dictionary(
    dictionary_type: str,
    n: int,
    n_atoms: int = None,
) -> None:
    if dictionary_type == 'fourier':
        D = dft(n, scale='sqrtn')
    elif dictionary_type == 'learnable':
        pass
        # D = np.zeros((self.stalk_dim, self.n_atoms))
    else:
        pass
    return D


def main():
    print('Start performing sanity tests...')

    agents = {
        0: {
            'model': 'vit_small_patch16_224',
            'dataset': 'cifar10',
            'seed': 42,
        },
        1: {
            'model': 'vit_small_patch32_224',
            'dataset': 'cifar10',
            'seed': 42,
        },
    }
    coder_params = {
        'explained_variance': 0.9,
        'sparsity_level': 0.1,
        'regularizer': 1e-3,
        'n_atoms': None,
        'dictionary_type': 'fourier',
        'max_iter': 100,
        'tol': 1e-2,
    }

    # =============================================================
    #                 TEST LATENTS' COMPRESSIBILITY
    # =============================================================
    agent = Agent(
        id=0,
        model=agents[0]['model'],
        dataset=agents[0]['dataset'],
        seed=agents[0]['seed'],
    )
    X, _ = agent.get_latent(prewhite=True, out='numpy')

    print('Learning the sparse representation...', end='\t')
    coder = SparseCoder(
        X=X,
        params=coder_params,
    )
    coder.fit()
    print(f'...{coder.S.shape=}, {coder.D.shape=}...', end='\t')
    print('[Passed]')

    # =============================================================
    #                 TEST NETWORK INITIALIZATION
    # =============================================================
    print('Preparing the network...')
    Network(
        agents_info=agents,
        coder_params=coder_params,
        device='cpu',
    )
    print('[Passed]')

    return None


if __name__ == '__main__':
    main()
