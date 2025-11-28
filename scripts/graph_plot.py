import sys
import pickle
from pathlib import Path

sys.path.append(str(Path(sys.path[0]).parent))
from src import Network

import matplotlib.pyplot as plt
import igraph as ig
import numpy as np
import pandas as pd
from matplotlib import cm, colors as mcolors

from omegaconf import DictConfig
import hydra

# Global colormap + normalization (shared by all plots)
ACC_CMAP = cm.get_cmap('RdYlGn')
ACC_NORM = mcolors.Normalize(vmin=0.5, vmax=1.0)  # fixed 0.5–1.0 range


# def _prepare_net_visuals(
#     net: Network,
#     layout: str | None = None,
#     with_labels: bool = True,
#     n_clusters: int | str | None = None,
#     value_fontsize: int = 26,
#     max_node_size: float = 100.0,
# ):
#     """
#     Compute and attach all visual attributes to net.graph:
#     - edge weights & colors
#     - node colors (accuracy)
#     - node sizes (sparsity)
#     - grouping frames and marks (if n_clusters == 'auto')
#     """
#     # --- Edge weights & labels
#     edge_losses = net.get_edge_losses()
#     weights = [
#         edge_losses[tuple(sorted([e.source, e.target]))] for e in net.graph.es
#     ]
#     net.graph.es['weight'] = weights
#     net.graph.es['label'] = [f'{w:.2f}' for w in weights]

#     # --- Layout
#     if isinstance(layout, str):
#         layout = net.graph.layout(layout)

#     # --- Clustering / groups
#     if n_clusters is None:
#         mark_tuples = None
#         frame_colors = None
#         frame_widths = None
#     elif n_clusters == 'auto':
#         names = [a.model_name for a in net.agents.values()]
#         prefixes = [n.split('_')[0] for n in names]
#         unique_prefixes = sorted(set(prefixes))

#         # group nodes by prefix
#         mark_groups = [
#             [i for i, p in enumerate(prefixes) if p == pref]
#             for pref in unique_prefixes
#         ]

#         palette = cm.get_cmap('Set2', max(1, len(unique_prefixes)))
#         alpha = 0.7
#         mark_colors = [
#             (*palette(i)[:3], alpha) for i in range(len(unique_prefixes))
#         ]
#         mark_tuples = list(zip(mark_groups, mark_colors))

#         border_palette = cm.get_cmap('Dark2', max(1, len(unique_prefixes)))
#         frame_colors = [
#             (*border_palette(unique_prefixes.index(p))[:3], 1.0)
#             for p in prefixes
#         ]
#         frame_widths = [2.0] * len(prefixes)
#     else:
#         raise NotImplementedError("Only n_clusters=None or 'auto' supported")

#     # --- Node colors: accuracy in [0,1] → RdYlGn, normalized in [0.5,1]
#     agents_list = list(net.agents.values())
#     accs = np.array([float(a.acc) for a in agents_list], dtype=float)
#     accs = np.clip(accs, 0.0, 1.0)
#     net.graph.vs['color'] = [tuple(ACC_CMAP(ACC_NORM(a))) for a in accs]

#     # --- Node sizes: from sparsity / stalk_dim
#     try:
#         sparsities = np.array(
#             [float(a.sparsity) for a in agents_list], dtype=float
#         )
#     except TypeError:
#         sparsities = np.array(
#             [float(a.stalk_dim) for a in agents_list], dtype=float
#         )

#     sp_min, sp_max = np.min(sparsities), np.max(sparsities)
#     denom = (sp_max - sp_min) if (sp_max - sp_min) > 0 else 1.0

#     # scale sizes using max_node_size from config
#     min_size = max_node_size * 0.4
#     max_size = max_node_size
#     net.graph.vs['size'] = (
#         min_size + ((sparsities - sp_min) / denom) * (max_size - min_size)
#     ).tolist()

#     # --- Edge colors lighter for weaker edges
#     w = np.asarray(weights, dtype=float)
#     norm_w = (w - w.min()) / (w.max() - w.min() + 1e-12)
#     greys = cm.get_cmap('Greys')
#     grey_vals = 0.9 - 0.6 * norm_w
#     net.graph.es['color'] = [tuple(greys(float(v))) for v in grey_vals]

#     # --- Labels
#     if with_labels:
#         net.graph.vs['label'] = [str(i) for i in range(len(net.graph.vs))]
#         net.graph.vs['label_color'] = 'black'
#         net.graph.vs['label_dist'] = 0
#     else:
#         net.graph.vs['label'] = None

#     # value_fontsize used for node & edge labels
#     net.graph.vs['label_size'] = value_fontsize
#     net.graph.es['label_size'] = value_fontsize

#     return layout, mark_tuples, frame_colors, frame_widths


def _prepare_net_visuals(
    net: Network,
    layout: str | None = None,
    with_labels: bool = True,
    n_clusters: int | str | None = None,
    cutting_threshold: float = None,
    n_edges: int = None,
    value_fontsize: int = 26,
    max_node_size: float = 100.0,
):
    """
    Compute and attach all visual attributes to net.graph:
    - edge weights & colors
    - node colors (accuracy)
    - node sizes (sparsity)
    - grouping frames and marks

    If n_clusters == 'auto':
        - run Louvain (community_multilevel) 500 times with UNWEIGHTED modularity
        - build a co-association matrix of vertex co-membership
        - build a consensus graph from pairs that co-occur >= 50% of runs
        - run Louvain again on this consensus graph to get a consensus / stochastic membership
        - use that clustering for mark_tuples, frame_colors, frame_widths
    """
    # --- Edge weights & labels
    edge_losses = net.get_edge_losses()

    if cutting_threshold is None:
        if n_edges is None:
            n_edges = int(net.n_agents * (net.n_agents - 1) / 2)
        assert (n_edges > 0) and (n_edges <= len(net.graph.get_edgelist())), (
            'n_edges must be a positive integer, smaller than the current number of edges in the graph.'
        )
        to_remove = list(
            dict(sorted(edge_losses.items(), key=lambda item: item[1])).keys()
        )[n_edges:]

    else:
        to_remove = [
            edge
            for edge, loss in edge_losses.items()
            if loss > cutting_threshold
        ]
    eids = net.graph.get_eids(to_remove, directed=False, error=False)
    eids = [eid for eid in eids if eid >= 0]
    net.graph.delete_edges(eids)

    weights = [
        edge_losses[tuple(sorted([e.source, e.target]))] for e in net.graph.es
    ]
    net.graph.es['weight'] = weights
    net.graph.es['label'] = [f'{w:.2f}' for w in weights]

    # --- Layout
    if isinstance(layout, str):
        layout = net.graph.layout(layout)

    # --- Clustering / groups: Louvain with stochastic consensus
    if n_clusters is None:
        mark_tuples = None
        frame_colors = None
        frame_widths = None

    elif n_clusters == 'auto':
        n_vertices = len(net.graph.vs)
        n_runs = 500

        # 1) Run Louvain 500 times with UNWEIGHTED modularity
        memberships = []
        for _ in range(n_runs):
            clustering = net.graph.community_multilevel(weights=None)
            memberships.append(clustering.membership)

        memberships = np.asarray(
            memberships, dtype=int
        )  # (n_runs, n_vertices)

        # 2) Co-association matrix: how often i and j are in the same community
        coassoc = np.zeros((n_vertices, n_vertices), dtype=float)
        for run_idx in range(n_runs):
            memb = memberships[run_idx]
            # group indices by community id
            for comm_id in np.unique(memb):
                idx = np.where(memb == comm_id)[0]
                if len(idx) <= 1:
                    continue
                coassoc[np.ix_(idx, idx)] += 1.0

        # don't care about diagonal, but keep it harmless
        np.fill_diagonal(coassoc, 0.0)

        # 3) Build consensus graph:
        #    connect i,j if they are together in >= 50% of the runs
        edges = []
        for i in range(n_vertices):
            for j in range(i + 1, n_vertices):
                if coassoc[i, j] >= 0.9 * n_runs:
                    edges.append((i, j))

        # if consensus graph ends up empty (degenerate), fall back to a single run
        if len(edges) == 0:
            # fall back: just one Louvain run, unweighted
            edge_sim = net.graph.es['weight']
            final_clustering = net.graph.community_multilevel(weights=edge_sim)
            membership = final_clustering.membership
        else:
            cons_graph = ig.Graph(n=n_vertices, edges=edges)
            # 4) Consensus / stochastic membership via Louvain on consensus graph, UNWEIGHTED
            final_clustering = cons_graph.community_multilevel(weights=None)
            membership = final_clustering.membership

        # Build mark groups from final membership
        communities = {}
        for v_idx, comm_id in enumerate(membership):
            communities.setdefault(comm_id, []).append(v_idx)
        mark_groups = list(communities.values())
        n_comms = len(mark_groups)

        # Colors for group "marks" (filled areas)
        palette = cm.get_cmap('Set2', max(1, n_comms))
        alpha = 0.7
        mark_colors = [(*palette(i)[:3], alpha) for i in range(n_comms)]
        mark_tuples = list(zip(mark_groups, mark_colors))

        # Frame colors for each node, according to its community
        border_palette = cm.get_cmap('Dark2', max(1, n_comms))
        frame_colors = [
            (*border_palette(membership[i])[:3], 1.0)
            for i in range(n_vertices)
        ]
        frame_widths = [2.0] * n_vertices

    else:
        raise NotImplementedError("Only n_clusters=None or 'auto' supported")

    # --- Node colors: accuracy in [0,1] → RdYlGn, normalized in [0.5,1]
    agents_list = list(net.agents.values())
    accs = np.array([float(a.acc) for a in agents_list], dtype=float)
    accs = np.clip(accs, 0.0, 1.0)
    net.graph.vs['color'] = [tuple(ACC_CMAP(ACC_NORM(a))) for a in accs]

    # --- Node sizes: from sparsity / stalk_dim
    try:
        sparsities = np.array(
            [float(a.sparsity) for a in agents_list], dtype=float
        )
    except TypeError:
        sparsities = np.array(
            [float(a.stalk_dim) for a in agents_list], dtype=float
        )

    sp_min, sp_max = np.min(sparsities), np.max(sparsities)
    denom = (sp_max - sp_min) if (sp_max - sp_min) > 0 else 1.0

    # scale sizes using max_node_size from config
    min_size = max_node_size * 0.4
    max_size = max_node_size
    net.graph.vs['size'] = (
        min_size + ((sparsities - sp_min) / denom) * (max_size - min_size)
    ).tolist()

    # --- Edge colors lighter for weaker edges
    w = np.asarray(weights, dtype=float)
    norm_w = (w - w.min()) / (w.max() - w.min() + 1e-12)
    greys = cm.get_cmap('Greys')
    grey_vals = 0.9 - 0.6 * norm_w
    net.graph.es['color'] = [tuple(greys(float(v))) for v in grey_vals]

    # --- Labels
    if with_labels:
        net.graph.vs['label'] = [str(i) for i in range(len(net.graph.vs))]
        net.graph.vs['label_color'] = 'black'
        net.graph.vs['label_dist'] = 0
    else:
        net.graph.vs['label'] = None

    # value_fontsize used for node & edge labels
    net.graph.vs['label_size'] = value_fontsize
    net.graph.es['label_size'] = value_fontsize

    df = pd.DataFrame(
        [
            [i, a.model_name, int(sparsities[i]), accs[i]]
            for i, a in enumerate(agents_list)
        ],
        columns=['idx', 'model', 'sparsity', 'acc'],
    )

    print(df.to_string(index=False, float_format='%.2f'))

    return layout, mark_tuples, frame_colors, frame_widths


def sheaf_plot_single(
    net,
    ax,
    layout: str | None = None,
    with_labels: bool = True,
    n_clusters: int | str | None = None,
    threshold: float = None,
    n_edges: int = None,
    value_fontsize: int = 26,
    max_node_size: float = 100.0,
):
    """
    Draw ONE sheaf graph into a given Matplotlib axis (ax), WITHOUT colorbar.
    """
    layout, mark_tuples, frame_colors, frame_widths = _prepare_net_visuals(
        net,
        layout=layout,
        with_labels=with_labels,
        n_clusters=n_clusters,
        n_edges=n_edges,
        cutting_threshold=threshold,
        value_fontsize=value_fontsize,
        max_node_size=max_node_size,
    )

    ig.plot(
        net.graph,
        target=ax,
        layout=layout,
        vertex_color=net.graph.vs['color'],
        vertex_size=net.graph.vs['size'],
        vertex_label=net.graph.vs['label'],
        vertex_frame_color=frame_colors,
        vertex_frame_width=frame_widths,
        edge_color=net.graph.es['color'],
        edge_label=net.graph.es['label'],
        mark_groups=mark_tuples,
        backend='matplotlib',
        bbox=(300, 300),
        margin=10,  # keep inner plot margin small; grid spacing is set separately
    )

    # No ticks, no axes
    ax.tick_params(
        left=False, bottom=False, labelleft=False, labelbottom=False
    )
    ax.set_frame_on(False)
    plt.show()


def sheaf_plot_grid(
    nets,
    layouts: list[str | None] | None = None,
    with_labels: bool = True,
    value_fontsize: int = 26,
    axis_fontsize: int = 20,
    threshold: float = None,
    n_edges: int = None,
    max_node_size: float = 100.0,
    margin: float = 0.02,
):
    """
    Create a 2x2 grid of sheaf plots (4 nets) plus a SINGLE shared colorbar.

    margin: fractional spacing between subplots (passed to wspace/hspace).
    """
    if len(nets) != 4:
        raise ValueError('sheaf_plot_grid expects exactly 4 nets.')

    if layouts is None:
        layouts = [None] * 4
    elif len(layouts) != 4:
        raise ValueError('layouts must be length 4 or None.')

    # Create figure with configurable spacing between subplots
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(16, 11),
        constrained_layout=False,
        gridspec_kw={'wspace': margin, 'hspace': margin},
    )

    axes = axes.flatten()

    # Draw each net in its subplot
    for i, (net, ax, lay) in enumerate(zip(nets, axes, layouts)):
        # n_clusters = 'auto' if i % 2 == 0 else None
        n_clusters = 'auto'
        sheaf_plot_single(
            net,
            ax=ax,
            layout=lay,
            with_labels=with_labels,
            n_clusters=n_clusters,
            threshold=threshold,
            n_edges=n_edges,
            value_fontsize=value_fontsize,
            max_node_size=max_node_size,
        )

    # Shared colorbar
    sm = cm.ScalarMappable(norm=ACC_NORM, cmap=ACC_CMAP)
    sm.set_array([])

    cax = fig.add_axes([0.92, 0.25, 0.015, 0.5])
    cbar = fig.colorbar(sm, cax=cax, orientation='vertical')
    cbar.set_label('Avg. Accuracy', fontsize=axis_fontsize)
    cbar.set_ticks([0.5, 0.75, 1.0])
    # value_fontsize also used for colorbar tick labels
    cbar.ax.tick_params(labelsize=value_fontsize, length=3)
    cbar.outline.set_visible(False)

    fig.tight_layout(rect=[0.0, 0.0, 0.9, 1.0])  # leave room on right for cbar

    return fig, axes


@hydra.main(
    config_path='../.conf/hydra/visualization',
    config_name='graph',
    version_base='1.3',
)
def main(cfg: DictConfig):
    base_dir = Path('.')

    graphs_path = base_dir / cfg.graphs_pickle

    with open(graphs_path, 'rb') as f:
        nets = pickle.load(f)

    nets = [nets[-1]]
    # print(len(nets))
    if len(nets) == 1:
        fig, ax = plt.subplots(
            1,
            1,
            figsize=(16, 11),
            constrained_layout=False,
            gridspec_kw={'wspace': cfg.margin, 'hspace': cfg.margin},
        )
        sheaf_plot_single(
            nets[0],
            ax,
            layout=cfg.layout,
            with_labels=True,
            n_clusters='auto',
            threshold=cfg.threshold,
            n_edges=cfg.n_edges,
            value_fontsize=cfg.value_fontsize,
            max_node_size=cfg.max_node_size,
        )
    else:
        fig, _ = sheaf_plot_grid(
            nets,
            layouts=[cfg.layout] * 4,
            with_labels=True,
            value_fontsize=cfg.value_fontsize,
            axis_fontsize=cfg.axis_fontsize,
            threshold=cfg.threshold,
            n_edges=cfg.n_edges,
            max_node_size=cfg.max_node_size,
            margin=cfg.margin,
        )

    # This will be saved into Hydra's run directory
    fig.savefig(cfg.output_file, bbox_inches='tight')


if __name__ == '__main__':
    main()
