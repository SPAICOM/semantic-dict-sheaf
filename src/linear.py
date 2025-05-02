""""""

import torch
from typing import Any
from igraph import Graph
from tqdm.auto import tqdm


class Agent:
    """ """

    def __init__(
        self,
        id: int,
        model: str,
        stalk_dim: int,
        device: str = 'cpu',
    ):
        self.id: int = id
        self.model: str = model  # or directly the model
        self.stalk_dim: int = stalk_dim
        self.device: str = device

        # Other variables
        self.restriction_maps: dict[int, torch.Tensor] = {}

    def map_initialization(
        self, neighbour_id: int, edge_stalk_dim: int
    ) -> None:
        """ """
        self.restriction_maps[neighbour_id] = torch.zeros(
            edge_stalk_dim, self.stalk_dim
        ).to(self.device)
        return None


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
                stalk_dim=info['stalk_dim'],
                device=self.device,
            )

    def graph_initialization(
        self,
        edges: list[tuple[int, int]] = None,
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
        return None

    def edges_capacity(
        self,
        edges_stalks: dict[tuple[int, int], int] = None,
    ) -> None:
        """ """
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


def main():
    """The main loop."""
    print('Start performing sanity tests...')

    # Variables definition
    agents_info: dict[int, dict[str, Any]] = {
        0: {'model': 'vit_base_patch16_224', 'stalk_dim': 192},
        1: {'model': 'vit_base_patch16_224', 'stalk_dim': 192},
    }
    device: str = 'cpu'

    print('First Test...', end='\t')
    net = Network(agents_info=agents_info, device=device)
    net.graph_initialization()
    net.edges_capacity()
    print('[Passed]')

    return None


if __name__ == '__main__':
    main()
