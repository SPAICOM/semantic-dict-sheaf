""""""

from pathlib import Path
import sys

sys.path.append(str(Path(sys.path[0]).parent))

from torch.utils.data import TensorDataset, DataLoader
from pytorch_lightning import Trainer
from sklearn.cluster import KMeans
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm, colors as mcolors
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csgraph
from tqdm.auto import tqdm
import jax.numpy as jnp
from typing import Any
import igraph as ig
import numpy as np
import torch


# import matplotlib.patches as mpatches

# from matplotlib.colors import Normalize
from wandb import Image

from src.utils import (
    cross_cosine_similarity,
    layout_embedding,
    save_sheaf_plt,
    random_stiefel,
    convert_output,
    convert_input,
    create_proto,
    prewhiten,
    n_atoms,
)
from src.datamodules import DataModuleClassifier
from src.coder import SparseCoder, GlobalDict
from src.neural import Classifier


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
        self.device: str = device
        self.seed: int = seed
        self.id: int = id

        # ================================================================
        #                     Neural Model's Parameters
        # ================================================================
        self.model_name: str = model
        self.model: Classifier = None
        self.loss = None
        self.acc = 0.0

        # ================================================================
        #                 Agent's Latent Space and Maps
        # ================================================================
        self.datamodule: DataModuleClassifier = None
        self.dataset: str = dataset
        if testing:
            self.X_train = kwargs['X_train']
            self.X_test = None
            self.stalk_dim = self.X_train.shape[0]
        else:
            self.latent_initialization()
        self.S: torch.Tensor = None
        self.D: torch.Tensor = None
        self.sparsity: int = None
        self.restriction_maps: dict[int, torch.Tensor] = {}

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
        elif self.stalk_dim == edge_stalk_dim:
            self.restriction_maps[neighbour_id] = torch.eye(
                self.stalk_dim,
                device=self.device,
            )
        else:
            self.restriction_maps[neighbour_id] = torch.randn(
                edge_stalk_dim,
                self.stalk_dim,
                device=self.device,
            )

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

    def sparse_initialization(
        self,
        coder_params: dict[Any] = {},
    ) -> None:
        """ """
        X, _ = self.get_latent(prewhite=False, scale=True, out='numpy')
        coder = SparseCoder(
            X=X,
            params=coder_params,
        )
        self.D, self.S = coder.fit(out='torch')
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
        scale: bool = False,
        n_subsampling: int = None,
        subsampling_strategy: str = 'random',
        test: bool = False,
        seed: int = 42,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if test:
            if prewhite:
                X_tr, _, _, X_te = prewhiten(self.X_train, self.X_test)
            else:
                X_tr = self.X_train
                X_te = self.X_test
        else:
            if prewhite:
                X_tr, _, _ = prewhiten(self.X_train, self.X_test)
            else:
                X_tr = self.X_train
                X_te = None

        if scale:
            X_tr /= torch.norm(X_tr, p='fro')
            X_te = (
                X_te / torch.norm(X_te, p='fro') if X_te is not None else X_te
            )

        if n_subsampling is not None:
            if subsampling_strategy == 'random':
                torch.manual_seed(seed)
                X_tr = X_tr[:, torch.randperm(X_tr.size(1))]
                X_tr = X_tr[:, :n_subsampling]
            elif subsampling_strategy == 'proto':
                X_tr = create_proto(
                    X_tr,
                    n_proto=n_subsampling,
                    out='torch',
                )
            else:
                raise NotImplementedError(
                    f'Sampling strategy {subsampling_strategy} not implemented.'
                )
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
        self.task_accuracy: list[float, float] = [None, None]
        self.task_loss: list[float, float] = [None, None]
        self.alignment_loss: float = torch.inf

    # ================================================================
    #                 Edge Feature Retrieval Methods
    # ================================================================

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
        prewhite: bool = False,
        scale: bool = False,
        test: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        X_i, X_i_test = self.head.get_latent(
            prewhite=prewhite,
            scale=scale,
            test=test,
        )
        X_j, X_j_test = self.head.get_latent(
            prewhite=prewhite,
            scale=scale,
            test=test,
        )
        return X_i, X_j, X_i_test, X_j_test

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

    # ================================================================
    #                 Edge Feature Updating Methods
    # ================================================================

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

    def update_alignment_loss(
        self,
        loss: float,
    ) -> None:
        self.alignment_loss = loss
        return None

    def update_task_performances(
        self,
        acc: float,
        loss: float,
        reverse: bool = False,
    ) -> None:
        if reverse:
            self.task_accuracy[1] = acc
            self.task_loss[1] = loss
        else:
            self.task_accuracy[0] = acc
            self.task_loss[0] = loss
        return None

    # ================================================================
    #                 Edge Metrics Retrieval Methods
    # ================================================================
    def return_task_accs(self) -> dict[str, float]:
        return self.task_accuracy

    def return_task_losses(self) -> dict[str, float]:
        return self.task_loss

    def return_alignment_loss(self) -> dict[str, float]:
        return self.alignment_loss

    def return_metrics(self) -> tuple[float, float, float]:
        """Return the edge metrics."""
        return self.task_accuracy, self.task_loss, self.alignment_loss

    def evaluate_sparsity_pattern_similarity(
        self, method: str = 'cosine'
    ) -> float:
        """Evaluate the similarity between the sparsity patterns of the two agents' sparse representations."""
        head_norms = torch.norm(self.head.S, p=2, dim=1)
        tail_norms = torch.norm(self.tail.S, p=2, dim=1)
        if method == 'cosine':
            similarity = torch.nn.functional.cosine_similarity(
                head_norms, tail_norms, dim=0
            )
        else:
            raise NotImplementedError(
                f'Similarity method {method} not implemented.'
            )
        return similarity.item()


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
        run=None,
    ):
        self.edges: dict[tuple[int, int], Edge] = {}
        self.agents: dict[int, Agent] = {}
        self.is_connection_graph: bool = False
        self.dictionary = dictionary
        self.device: str = device
        self.global_dim: int = 0
        # self.globalDict: torch.Tensor = None
        self.n_agents: int = 0
        self.n_edges: int = 0
        self.run = run

        self.coder_params: dict[Any] = {
            'prewhite': False,
            'scale': False,
            'n_subsampling': None,
            'sampling_strategy': 'random',
        }
        self.coder_params.update(coder_params)

        # Init Agents
        X_train_dict = []
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
            X, _ = self.agents[idx].get_latent(
                prewhite=self.coder_params['prewhite'],
                scale=self.coder_params['scale'],
                n_subsampling=self.coder_params['n_subsampling'],
                subsampling_strategy=self.coder_params['sampling_strategy'],
                out='numpy',
            )
            X_train_dict.append(X)

        self.n_agents = len(self.agents)
        X_train_dict = np.hstack(X_train_dict)

        self.set_global_dictionary(
            X_train_dict,
            dict_params=self.coder_params,
        )

        # Init Edges
        if edges_info is None:
            self.graph = ig.Graph.Full(n=self.n_agents)
        else:
            self.graph = ig.Graph()
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
            self.edges[(i, j)] = Edge(
                head=self.agents[i],
                tail=self.agents[j],
                stalk_dim=e_stalk,
                mask=mask_edges,
                device=self.device,
            )
        self.n_edges = len(self.edges)
        self._is_connection_graph()
        del agents_info, edges_info

    def set_global_dictionary(
        self,
        X_train: np.ndarray,
        dict_params: dict[Any] = {},
    ) -> None:
        GD = GlobalDict(
            X=X_train,
            agents=self.agents,
            n_nodes=self.n_agents,
            params=dict_params,
            run=self.run,
        )
        DD, SS = GD.fit(out='torch')

        for i in range(self.n_agents):
            self.agents[i].S = SS[i]
            self.agents[i].sparsity = n_atoms(self.agents[i].S)
            self.agents[i].D = DD[i]

        self.dict_metrics = GD.return_metrics()

        return None

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

    def get_edge_losses(self) -> dict[tuple[int, int], float]:
        """Return a dictionary with the edge ids as keys and the corresponding edge losses as values."""
        return {
            edge.id: edge.return_alignment_loss()
            for edge in self.edges.values()
        }

    def update_graph(
        self,
        n_edges: int,
    ) -> None:
        """Update the graph based on the edge losses."""
        assert (n_edges > 0) and (n_edges <= len(self.graph.get_edgelist())), (
            'n_edges must be a positive integer, smaller than the current number of edges in the graph.'
        )

        edge_losses = self.get_edge_losses()
        to_remove = list(
            dict(sorted(edge_losses.items(), key=lambda item: item[1])).keys()
        )[n_edges:]
        eids = self.graph.get_eids(to_remove, directed=False, error=False)
        eids = [eid for eid in eids if eid >= 0]
        self.graph.delete_edges(eids)
        self.n_edges = n_edges

    def prepare_message(
        self,
        tx_id: int,
    ) -> torch.Tensor:
        """Prepare the message to be sent from tx_id to rx_id."""

        _, X_te = self.agents[tx_id].get_latent(
            prewhite=False,
            scale=True,
            test=True,
            out='numpy',
        )
        coder = SparseCoder(
            X=X_te,
            dict_type='pca',
            params=self.coder_params,
        )
        _, S = coder.fit(out='torch')
        message = self.agents[tx_id].D @ S
        return message

    def send_message(
        self,
        tx_id: int,
        rx_id: int,
    ) -> torch.Tensor:
        """Send data from tx_id to rx_id through the edge connecting them."""

        # message = self.prepare_message(tx_id=tx_id)

        try:
            mask = self.edges[(tx_id, rx_id)].mask
        except KeyError:
            mask = self.edges[(rx_id, tx_id)].mask

        # print(f'{self.agents[tx_id].D.shape=}')
        # print(f'{self.agents[rx_id].restriction_maps[tx_id].T.shape=}')
        # print(f'{self.agents[tx_id].restriction_maps[rx_id].shape=}')
        # print(f'{message.shape=}')

        message = (
            (
                self.agents[rx_id].restriction_maps[tx_id].T
                @ self.agents[tx_id].restriction_maps[rx_id]
                @ self.agents[tx_id].X_test
            )
            if mask is None
            else (
                self.agents[rx_id].restriction_maps[tx_id].T
                @ mask
                @ self.agents[tx_id].restriction_maps[rx_id]
                @ self.agents[tx_id].X_test
            )
        )

        return message

    def test_communication(
        self,
        rx_id: int,
        tx_id: int,
        trainer: Trainer,
    ) -> tuple[float, float]:
        """Evaluate accuracy and loss for a specific agent (receiver agent) using latent representations
        from one of its neighbors (sender agent) in the inferred network sheaf. The considered latent
        representations are aligned by the learned restriction map to the corresponding edge between the two agents.
        """
        # Send the latent representations
        rx_data = self.send_message(
            tx_id=tx_id,
            rx_id=rx_id,
        )
        # Evaluate the receiver agent model on the received data
        self.agents[rx_id].load_model(
            model_path=f'models/classifiers/{self.agents[rx_id].dataset}/{self.agents[rx_id].model_name}/seed_{self.agents[rx_id].seed}.ckpt'
        )
        accuracy, loss = self.agents[rx_id].test_model(
            data=rx_data,
            trainer=trainer,
        )

        # Update the edge task performances
        print((rx_id, tx_id), accuracy, loss)
        if (tx_id, rx_id) in self.edges:
            edge = (tx_id, rx_id)
            reverse = False
        else:
            edge = (rx_id, tx_id)
            reverse = True
        self.edges[edge].update_task_performances(
            accuracy,
            loss,
            reverse=reverse,
        )

        return accuracy, loss

    def test_agent_model(
        self,
        agent_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate average accuracy and loss for a specific agent using latent representations
        from its neighbors in the inferred network sheaf. The considered latent representations
        are the ones obtained by sending messages from the neighbors to the agent,
        aligned by the corresponding learned restriction maps.
        """
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
            acc, loss = self.test_communication(
                rx_id=agent_id,
                tx_id=rx_id,
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
        verbose: bool = False,
    ) -> None:
        """ """
        for i, agent in self.agents.items():
            acc, loss = self.test_agent_model(agent_id=i)
            if verbose:
                print(
                    f'Agent-{i} ({agent.model_name}) - Task Accuracy (Test): {acc}'
                )
                print(f'Agent-{i} ({agent.model_name}) - Loss (Test): {loss}')
            if self.run is not None:
                self.run.log(
                    {
                        f'Agent-{i} ({agent.model_name}) - Task Accuracy (Test)': acc,
                        f'Agent-{i} ({agent.model_name}) - Loss (Test)': loss,
                    }
                )
            agent.acc, agent.loss = (acc, loss)
        return None

    def return_alignment_metrics(self) -> dict[str, float]:
        """Compute alignment loss for network edges.
        Report edge-level metrics for alignment and task performance comparison.
        """

        metrics = (
            {
                'edge_id': [],
                'sparsity_pattern_similarity': [],
                'alignment_loss': [],
                'task_accuracy': [],
                'task_loss': [],
                'task_accuracy_rev': [],
                'task_loss_rev': [],
            }
            if self.coder_params['dict_type'] == 'learnable'
            else {
                'edge_id': [],
                'alignment_loss': [],
                'task_accuracy': [],
                'task_loss': [],
                'task_accuracy_rev': [],
                'task_loss_rev': [],
            }
        )
        for edge in self.edges.values():
            metrics['edge_id'].append(edge.id)
            if self.coder_params['dict_type'] == 'learnable':
                metrics['sparsity_pattern_similarity'].append(
                    edge.evaluate_sparsity_pattern_similarity()
                )
            metrics['alignment_loss'].append(edge.return_alignment_loss())
            accs = edge.return_task_accs()
            losses = edge.return_task_losses()
            metrics['task_accuracy'].append(accs[0])
            metrics['task_loss'].append(losses[0])
            metrics['task_accuracy_rev'].append(accs[1])
            metrics['task_loss_rev'].append(losses[1])

        return metrics

    def return_dict_metrics(self) -> dict[str, float]:
        """Compute the dictionary metrics for each agent.
        Report agent-level metrics for sparse coding and task performance comparison.
        """
        if 'acc' not in self.dict_metrics:
            metrics = {
                'agent_id': self.dict_metrics['agent_id'],
                'sparsity': self.dict_metrics['sparsity'],
                'nmse': self.dict_metrics['nmse'],
                'acc': [],
            }
            for agent in self.agents.values():
                metrics['acc'].append(agent.acc)
            self.dict_metrics = metrics
        return self.dict_metrics

    @save_sheaf_plt
    def sheaf_plot(
        self,
        layout: str = None,
        with_labels: bool = True,
        n_clusters: int = None,
        seed: int = 42,
    ) -> None:
        """Plot the sheaf with a slim right colorbar, node numbers inside nodes, and a table below."""
        # --- Edge weights & labels
        edge_losses = self.get_edge_losses()
        weights = [
            edge_losses[tuple(sorted([e.source, e.target]))]
            for e in self.graph.es
        ]
        self.graph.es['weight'] = weights
        self.graph.es['label'] = [f'{w:.2f}' for w in weights]

        # --- Layout
        if layout is None:
            layout, _ = layout_embedding(
                graph=self.graph, layout=layout, seed=seed
            )
        else:
            layout = self.graph.layout(layout)

        # --- Optional spectral clustering
        if n_clusters is None:
            mark_groups = None
        else:
            n = len(self.graph.vs)
            A = np.zeros((n, n), dtype=float)
            for e in self.graph.es:
                A[e.source, e.target] = A[e.target, e.source] = e['weight']
            L = csgraph.laplacian(A, normed=False)
            _, eigvecs = np.linalg.eigh(L)
            X = eigvecs[:, 1 : n_clusters + 1]
            kmeans = KMeans(n_clusters=n_clusters, random_state=seed).fit(X)
            cluster_labels = kmeans.labels_.tolist()
            self.graph.vs['cluster'] = cluster_labels
            mark_groups = [[] for _ in range(n_clusters)]
            for idx, label in enumerate(cluster_labels):
                mark_groups[label].append(idx)

        # --- Node colors from accuracy [0,1] using RdYlGn
        agents_list = list(
            self.agents.values()
        )  # assumed aligned with vertex order
        accs = np.array([float(a.acc) for a in agents_list], dtype=float)
        accs = np.clip(accs, 0.0, 1.0)
        cmap = cm.get_cmap('RdYlGn')
        acc_norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
        self.graph.vs['color'] = [tuple(cmap(acc_norm(a))) for a in accs]

        # --- Node sizes: sparsity → larger range so numbers fit inside nodes
        sparsities = np.array(
            [float(a.sparsity) for a in agents_list], dtype=float
        )
        sp_min, sp_max = np.min(sparsities), np.max(sparsities)
        denom = (sp_max - sp_min) if (sp_max - sp_min) > 0 else 1.0
        min_size, max_size = 26.0, 72.0  # enlarged range
        self.graph.vs['size'] = (
            min_size + ((sparsities - sp_min) / denom) * (max_size - min_size)
        ).tolist()

        # --- Edge colors: lighter for weaker edges
        w = np.asarray(weights, dtype=float)
        norm_w = (w - w.min()) / (w.max() - w.min() + 1e-12)
        greys = cm.get_cmap('Greys')
        grey_vals = 0.9 - 0.6 * norm_w
        self.graph.es['color'] = [tuple(greys(float(v))) for v in grey_vals]

        # --- Vertex labels: ONLY node numbers, centered; color depends on accuracy
        if with_labels:
            self.graph.vs['label'] = [
                str(i) for i in range(len(self.graph.vs))
            ]
            # White text except when 0.3 <= acc <= 0.7 -> black (for yellow-ish nodes)
            label_colors = [
                'black' if 0.3 <= float(a) <= 0.7 else 'white' for a in accs
            ]
            self.graph.vs['label_color'] = label_colors  # per-vertex attribute
            self.graph.vs['label_size'] = 14  # font size of numbers
            self.graph.vs['label_dist'] = 0  # center inside nodes
        else:
            self.graph.vs['label'] = None

        # --- Figure: graph on top, table below
        fig = plt.figure(figsize=(16, 11))
        gs = fig.add_gridspec(
            nrows=2, ncols=1, height_ratios=[0.78, 0.22], hspace=0.10
        )
        ax_graph = fig.add_subplot(gs[0, 0])
        ax_table = fig.add_subplot(gs[1, 0])

        # --- Draw graph (no titles)
        ig.plot(
            self.graph,
            target=ax_graph,
            layout=layout,
            vertex_color=self.graph.vs['color'],
            vertex_size=self.graph.vs['size'],
            vertex_label=self.graph.vs['label'],
            # DO NOT pass vertex_label_color here — use per-vertex attribute set above
            edge_color=self.graph.es['color'],
            edge_label=self.graph.es['label'],
            mark_groups=mark_groups,
            bbox=(300, 300),
            margin=40,
        )

        # --- Slim vertical colorbar on the RIGHT of the graph
        sm = cm.ScalarMappable(norm=acc_norm, cmap=cmap)
        sm.set_array([])
        divider = make_axes_locatable(ax_graph)
        cax = divider.append_axes(
            'right', size='0.6%', pad=0.02
        )  # small & tidy
        cbar = fig.colorbar(sm, cax=cax, orientation='vertical')
        cbar.set_label('Avg. Accuracy', fontsize=13, rotation=90, labelpad=10)
        cbar.set_ticks([0.0, 0.5, 1.0])
        cbar.ax.tick_params(labelsize=12, length=3)  # larger tick numbers
        cbar.outline.set_visible(False)

        # Hide graph ticks
        ax_graph.tick_params(
            left=False, bottom=False, labelleft=False, labelbottom=False
        )

        # --- Table below the graph
        ax_table.axis('off')
        rows = [
            [i, a.model_name, int(sparsities[i]), f'{accs[i]:.2f}']
            for i, a in enumerate(agents_list)
        ]
        table = ax_table.table(
            cellText=rows,
            colLabels=['Node', 'Model', 'Dimension', 'Accuracy'],
            cellLoc='center',
            colLoc='center',
            loc='center',
        )
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.05, 1.18)

        plt.tight_layout()
        self.run.log({'final_network': Image(fig)})
        return None

    def restriction_maps_heatmap(
        self,
        n: int,
    ) -> None:
        """Plot the restriction maps for the best n communication edges."""
        cols = n // 2
        fig, axes = plt.subplots(2, cols, figsize=(10, 5))

        for ax, edge in zip(axes.flatten(), list(self.edges.values())):
            _, F_j = edge.get_restriction_maps(out='numpy')

            # Min–max normalize to [0, 1] per heatmap
            mn = np.nanmin(F_j)
            mx = np.nanmax(F_j)
            if not np.isfinite(mn) or not np.isfinite(mx) or mx - mn == 0:
                F_j = np.zeros_like(F_j, dtype=float)
            else:
                F_j = (F_j - mn) / (mx - mn)

            sns.heatmap(
                F_j,
                ax=ax,
                cmap='viridis',
                cbar_kws={'label': 'Value'},
            )
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f'Restriction Map on Edge {edge.id}')

        plt.tight_layout()
        self.run.log({'restriction_maps': Image(plt)})
        return None

    def pca_correlation_heatmap(
        self,
        k: int,
        n: int,
    ) -> None:
        """Plot the similarity of the first k principal components of two agents for the best k communication edges."""
        cols = n // 2
        fig, axes = plt.subplots(2, cols, figsize=(10, 5))
        ax_list = axes.ravel()
        for ax, edge in zip(ax_list, list(self.edges.values())):
            D_i, D_j = edge.get_dictionaries(out='numpy')
            Sim = cross_cosine_similarity(D_i, D_j, k)
            sns.heatmap(
                Sim,
                ax=ax,
                cmap='viridis',
                cbar_kws={'label': 'Value'},
            )
            ax.set_xlabel('Principal Component')
            ax.set_ylabel('Principal Component')
            ax.set_title(f'Cross-Covariance on Edge {edge.id}')

        plt.tight_layout()
        self.run.log({'pca_correlation': Image(plt)})
        return None


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
        'dict_type': None,
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
    X, _ = agent.get_latent(scale=True, prewhite=True, test=True, out='numpy')

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
    coder_params = {
        'init_mode': 'random',
        'regularizer': 1e-3,
        'dict_type': None,
        'n_atoms': 400,
        'max_iter': 100,
        'Dstep': 1,
        'Sstep': 1,
        'tol': 1e-3,
    }
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
