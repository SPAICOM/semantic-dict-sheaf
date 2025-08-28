""""""

from pathlib import Path
import sys

sys.path.append(str(Path(sys.path[0]).parent))

from torch.utils.data import TensorDataset, DataLoader
from pytorch_lightning import Trainer
from typing import Any
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from matplotlib import cm
import jax.numpy as jnp
import igraph as ig
import numpy as np
import torch


# import matplotlib.patches as mpatches

# from matplotlib.colors import Normalize
from scipy.sparse import csgraph
from wandb import Image

from src.utils import (
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

        metrics = {
            'edge_id': [],
            'sparsity_pattern_similarity': [],
            'alignment_loss': [],
            'task_accuracy': [],
            'task_loss': [],
            'task_accuracy_rev': [],
            'task_loss_rev': [],
        }
        for edge in self.edges.values():
            metrics['edge_id'].append(edge.id)
            metrics['sparsity_pattern_similarity'].append(
                edge.evaluate_sparsity_pattern_similarity()
            )  # TODO
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
        return self.dict_metrics

    @save_sheaf_plt
    def sheaf_plot(
        self,
        layout: list[tuple[int]] = None,
        with_labels: bool = True,
        n_clusters: int = 2,
        seed: int = 42,
    ) -> None:
        """Plot the sheaf with spectral clustering and custom aesthetics using igraph."""

        # Edge weights and labels
        edge_losses = self.get_edge_losses()
        weights = [
            edge_losses[tuple(sorted([e.source, e.target]))]
            for e in self.graph.es
        ]
        self.graph.es['weight'] = weights
        self.graph.es['label'] = [f'{w:.2f}' for w in weights]

        # Layout
        layout, _ = layout_embedding(
            graph=self.graph, layout=layout, seed=seed
        )

        # Spectral clustering using graph Laplacian
        n = len(self.graph.vs)
        A = np.zeros((n, n))
        for e in self.graph.es:
            A[e.source, e.target] = A[e.target, e.source] = e['weight']

        L = csgraph.laplacian(A, normed=False)
        _, eigvecs = np.linalg.eigh(L)
        X = eigvecs[:, 1 : n_clusters + 1]
        kmeans = KMeans(n_clusters=n_clusters, random_state=seed).fit(X)
        cluster_labels = kmeans.labels_.tolist()  # Convert to Python list
        self.graph.vs['cluster'] = cluster_labels

        # Node color based on accuracy, mapped to [0.5, 1.0] in RdYlGn colormap
        accs = np.array([agent.acc for agent in self.agents.values()])
        acc_norm = (accs - accs.min()) / (np.ptp(accs) + 1e-6)
        acc_scaled = 0.5 + acc_norm * 0.5
        cmap = cm.get_cmap('RdYlGn')
        self.graph.vs['color'] = [cmap(a) for a in acc_scaled]

        # Node size based on sparsity
        min_size = 10
        max_size = 40
        sparsities = np.array(
            [agent.sparsity for agent in self.agents.values()]
        )
        sparsity_min = sparsities.min()
        sparsity_max = sparsities.max()
        sparsity_range = sparsity_max - sparsity_min + 1e-6
        normalized_sizes = min_size + (
            (sparsities - sparsity_min) / sparsity_range
        ) * (max_size - min_size)
        self.graph.vs['size'] = normalized_sizes.tolist()

        # Edge color in greyscale (to fake alpha fading)
        norm_weights = (np.array(weights) - min(weights)) / (
            max(weights) - min(weights) + 1e-5
        )
        greys = cm.get_cmap('Greys')
        self.graph.es['color'] = [
            greys(0.3 + 0.7 * (1 / w + 1e-6)) for w in norm_weights
        ]

        # Labels
        if with_labels:
            self.graph.vs['label'] = [
                f'{agent.id}: {agent.model_name}\nAcc: {agent.acc:.2f} - N. Atoms: {agent.sparsity}'
                for agent in self.agents.values()
            ]
        else:
            self.graph.vs['label'] = None

        # Clustering groups for `mark_groups`
        mark_groups = [[] for _ in range(n_clusters)]
        for idx, label in enumerate(cluster_labels):
            mark_groups[label].append(idx)

        # Plotting
        _, ax = plt.subplots(figsize=(20, 20))
        plt.title(f"Sheaf learned with '{self.run.name}' algorithm")

        ig.plot(
            self.graph,
            target=ax,
            layout=layout,
            vertex_color=self.graph.vs['color'],
            vertex_size=self.graph.vs['size'],
            vertex_label=self.graph.vs['label'],
            edge_color=self.graph.es['color'],
            edge_label=self.graph.es['label'],
            # mark_groups=mark_groups,
            # mark_color=[(0.8, 0.8, 0.8, 0.2)] * n_clusters,
            bbox=(300, 300),
            margin=40,
        )
        plt.tight_layout()
        self.run.log({'final_network': Image(plt)})
        # self.run.summary['final_network'] = Image(plt)


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
