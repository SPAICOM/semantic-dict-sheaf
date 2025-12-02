import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.patches import Rectangle
from matplotlib import rcParams
from plotnine import (
    ggplot,
    aes,
    geom_violin,
    geom_point,
    geom_line,
    geom_hline,
    geom_label,
    element_rect,
    element_blank,
    element_text,
    scale_fill_manual,
    scale_color_manual,
    scale_x_discrete,
    coord_flip,
    theme_classic,
    theme,
    stage,
)

import igraph as ig
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as mcolors
import numpy as np
from pathlib import Path


def graph_edge_loss(df: pl.DataFrame, th: float = 0.75) -> ig.Graph:
    # Filter on alignment_loss
    df = df.filter(pl.col('alignment_loss') <= th)

    edgelist = df['edge_id'].to_list()  # list of [u, v]
    weights = df['alignment_loss'].to_list()
    natures = df['Edge Nature'].to_list()

    g = ig.Graph(edges=edgelist, directed=False)

    g.es['weight'] = weights
    g.es['nature'] = natures
    g.es['label'] = natures

    return g


def homo_hetero(df: pl.DataFrame):
    """"""
    homo_edges = [
        [0, 1],
        [1, 2],
        [0, 2],
        [2, 3],
        [0, 3],
        [1, 3],
        [8, 9],
        [4, 7],
        [5, 7],
        [4, 5],
    ]
    df = df.with_columns(
        pl.when(pl.col('edge_id').is_in(homo_edges))
        .then(pl.lit('Homophilic'))
        .otherwise(pl.lit('Heterophilic'))
        .alias('Edge Nature')
    )
    return df


def graph_plot(
    g: ig.Graph,
    nodes_color: str = 'white',
    ax: plt.Axes | None = None,
    edge_color_map: dict | None = None,
):
    """
    Plot a graph with group shading & colored borders into a given Matplotlib Axes.
    If `ax` is None, a new figure/axes is created and returned.

    edge_color_map: dict like {'Homophilic': '#....', 'Heterophilic': '#....'}
                    If None, a default mapping is used.
    """
    model_names = [
        'vit_small_patch16_224',  # node 0
        'vit_small_patch16_384',  # node 1
        'vit_small_patch32_224',  # node 2
        'vit_small_patch32_384',  # node 3
        'levit_128',  # node 4
        'levit_192',  # node 5
        'efficientvit_m4',  # node 6
        'levit_conv_128',  # node 7
        'volo_d1_224',  # node 8
        'volo_d1_384',  # node 9
    ]

    prefixes = [n.split('_')[0] for n in model_names]
    unique_prefixes = sorted(set(prefixes))

    # --- group shading (Set2) -------------------------------------------------
    set2 = matplotlib.colormaps.get_cmap('Set2').resampled(
        max(1, len(unique_prefixes))
    )
    alpha = 0.9
    mark_groups = [
        [i for i, p in enumerate(prefixes) if p == pref]
        for pref in unique_prefixes
    ]
    mark_colors = [(*set2(i)[:3], alpha) for i in range(len(unique_prefixes))]
    mark_tuples = list(zip(mark_groups, mark_colors))

    # --- per-vertex border colors (Dark2) ------------------------------------
    dark2 = matplotlib.colormaps.get_cmap('Dark2').resampled(
        max(1, len(unique_prefixes))
    )
    frame_colors = [
        mcolors.to_hex(dark2(unique_prefixes.index(p))[:3]) for p in prefixes
    ]
    frame_widths = [2.0] * g.vcount()

    # --- edge colors from `nature` -------------------------------------------
    if edge_color_map is None:
        edge_color_map = {
            'Homophilic': '#1f77b4',
            'Heterophilic': '#ff7f0e',
        }

    if 'nature' in g.es.attribute_names():
        natures = g.es['nature']
        edge_colors = [edge_color_map.get(n, '#999999') for n in natures]
        g.es['label'] = natures
        g.es['color'] = edge_colors
    else:
        if 'label' not in g.es.attribute_names():
            g.es['label'] = ['' for _ in range(g.ecount())]
        g.es['color'] = ['#999999'] * g.ecount()

    # --- vertex color / size defaults ----------------------------------------
    if 'color' not in g.vs.attribute_names():
        g.vs['color'] = [nodes_color] * g.vcount()
    if 'size' not in g.vs.attribute_names():
        g.vs['size'] = [12] * g.vcount()

    # --- layout with extra padding to avoid cut shadows ----------------------
    # Use an explicit layout so we can compute bounds
    layout = g.layout('kk')
    coords = np.array(layout.coords, dtype=float)
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    span = max(x_max - x_min, y_max - y_min)
    pad = 0.3 * span

    # --- plot into given axes -------------------------------------------------
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    ig.plot(
        g,
        layout='kk',
        target=ax,
        vertex_color=g.vs['color'],
        vertex_size=g.vs['size'],
        # vertex_frame_color=frame_colors,
        # vertex_frame_width=frame_widths,
        edge_color=g.es['color'],
        mark_groups=mark_tuples,
        backend='matplotlib',
        bbox=(300, 400),
        margin=150,
    )

    ax.set_xlim(x_min - pad, x_max + pad)
    ax.set_ylim(y_min - pad, y_max + pad)

    ax.set_xticks([])
    ax.set_yticks([])
    # ax.set_frame_on(False)

    return ax


# Small helper for annotations on a given axis (using axes fraction coords)
def annotations(ax, text, xy, **kwargs):
    ax.annotate(
        text,
        xy=xy,
        xycoords='axes fraction',  # (0,0) bottom-left, (1,1) top-right of the axes
        **kwargs,
    )


def edges_distributions(df: pl.DataFrame):
    # --- global font settings -------------------------------------------------
    rcParams['mathtext.fontset'] = 'stix'
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Times New Roman', 'Times']

    df_pl = df

    # thresholds shared by central plot + 3+3 graphs
    thresholds = [0.6, 0.75, 0.88]
    labels = ['0.60', '0.75', '0.88']
    label_x = [0.60, 0.75, 0.88]

    label_df = pd.DataFrame(
        {
            'alignment_loss': thresholds,
            'label': labels,
            'label_x': label_x,
            'Case': ['Dict.'] * len(thresholds),
        }
    )

    # colors for threshold lines (and for the dashed boxes)
    line_colors = ['#1b1b7a', '#7a1b1b', '#1b7a1b']  # blue, red, green-ish
    label_shifts = [0.5, 0.8, 1.1]

    # --- color maps -----------------------------------------------------------
    # Case fill colors: NO DICT = blue, DICT = orange
    case_fill_map = {
        'No Dict.': 'dodgerblue',
        'Dict.': 'darkorange',
    }

    # Edge Nature colors (nice complementary-ish: green & purple)
    edge_color_map = {
        'Homophilic': '#2ca02c',  # green
        'Heterophilic': '#9467bd',  # purple
    }

    # --- pandas version for plotnine -----------------------------------------
    df_pd = df_pl.to_pandas()
    df_pd['edge_group'] = df_pd['edge_id'].apply(lambda e: tuple(e))

    lsize = 0.65
    fill_alpha = 0.7
    shift_violin = 0.05
    shift_point = 0.02

    # aesthetics with stage / after_scale
    m1 = aes(
        x=stage('Case', after_scale=f'x + {shift_violin}*((-1)**x)'),
        y='alignment_loss',
        fill='Case',
    )

    m2_point = aes(
        x=stage('Case', after_scale=f'x - {shift_point}*((-1)**x)'),
        y='alignment_loss',
        color='Edge Nature',
        group='edge_group',
    )

    # CHANGE 1: color lines by Edge Nature (same as the points)
    m2_line = aes(
        x=stage('Case', after_scale=f'x - {shift_point}*((-1)**x)'),
        y='alignment_loss',
        group='edge_group',
        color='Edge Nature',
    )

    base_size = 18

    # --- central plotnine figure ---------------------------------------------
    p = (
        ggplot(df_pd)
        + geom_violin(
            m1,
            style='left-right',
            alpha=fill_alpha,
            size=lsize,
            width=2,
        )
        # line color now mapped to Edge Nature via m2_line (no fixed gray color)
        + geom_line(m2_line, size=lsize - 0.2, alpha=0.6)
        + geom_point(m2_point, alpha=fill_alpha, size=4)
        + scale_x_discrete(expand=[0.12, 0])
        # Case colors (No Dict. = blue, Dict. = orange)
        + scale_fill_manual(
            name='Case',
            values=case_fill_map,
        )
        # Edge Nature colors (green/purple) shared with igraph edges
        + scale_color_manual(
            name='Edge Nature',
            values=edge_color_map,
        )
        + coord_flip()
        + theme_classic(base_size=base_size, base_family='Times New Roman')
        + theme(
            figure_size=(20, 10),  # tall for 3 rows
            panel_background=element_rect(fill='white'),
            plot_background=element_rect(fill='white'),
            panel_grid_major=element_blank(),
            panel_grid_minor=element_blank(),
            axis_title=element_blank(),
            axis_text=element_blank(),
            axis_ticks=element_blank(),
            axis_line=element_blank(),
            panel_border=element_blank(),
            text=element_text(family='Times New Roman', size=base_size),
            legend_position='none',
        )
    )

    # vertical lines + labels (inside central panel)
    for t, lbl, col, lx, shift in zip(
        thresholds, labels, line_colors, label_x, label_shifts
    ):
        p += geom_hline(
            yintercept=t,
            linetype='dashed',
            size=0.7,
            color=col,
        )

        row_df = label_df[label_df['label'] == lbl]

        p += geom_label(
            aes(
                x=stage(
                    'Case',
                    after_scale=f'x + {shift}*((-1)**x)',
                ),
                y='label_x',
                label='label',
            ),
            data=row_df,
            ha='center',
            va='center',
            size=base_size,
            color=col,
            fill='white',
            label_size=0.3,
        )

    # --- draw plotnine figure, get Matplotlib fig ----------------------------
    fig = p.draw()
    if not fig.axes:
        raise RuntimeError('No axes found in plotnine figure.')

    fig.set_tight_layout(False)  # we'll position axes manually

    ax_center = fig.axes[0]

    # central panel: large; graphs will be smaller above/below
    # [left, bottom, width, height]
    ax_center.set_position([0.08, 0.22, 0.64, 0.56])
    # y from 0.22 to 0.78

    # CHANGE 2: add annotations inside the central panel
    # Using axes-fraction coordinates so they stay inside the plot area.
    annotations(
        ax_center,
        'low edge loss',
        xy=(-0.01, 0.5),  # left-center
        ha='left',
        va='center',
        fontsize=base_size,
        fontstyle='italic',
        bbox=dict(
            boxstyle='round,pad=0.5',
            facecolor='white',
            edgecolor='none',
            alpha=0.9,
        ),
        zorder=10,
    )
    annotations(
        ax_center,
        'high edge loss',
        xy=(0.98, 0.5),  # right-center
        ha='right',
        va='center',
        fontsize=base_size,
        fontstyle='italic',
        bbox=dict(
            boxstyle='round,pad=0.5',
            facecolor='white',
            edgecolor='none',
            alpha=0.9,
        ),
        zorder=10,
    )

    # --- prepare data for graphs ---------------------------------------------
    no_dict_pl = df_pl.filter(pl.col('Case') == 'No Dict.')
    dict_pl = df_pl.filter(pl.col('Case') == 'Dict.')

    no_dict_color = case_fill_map['No Dict.']
    dict_color = case_fill_map['Dict.']

    no_dict_graphs = [
        no_dict_pl.pipe(graph_edge_loss, th=th) for th in thresholds
    ]
    dict_graphs = [dict_pl.pipe(graph_edge_loss, th=th) for th in thresholds]

    # --- layout for 3+3 small panels (smaller graphs) ------------------------
    # Make graphs quite small so distributions dominate
    top_bottom = 0.62  # bottom of top row (above central top ~0.78)
    bottom_bottom = 0.22  # bottom of bottom row (below central bottom ~0.22)
    row_height = 0.15  # small graphs

    left_margin = 0.26
    panel_width = 0.12
    hgap = 0.02

    no_dict_axes = []
    dict_axes = []

    # top row: 3 "No Dict" graphs
    for i, (th, g) in enumerate(zip(thresholds, no_dict_graphs)):
        left = left_margin + i * (panel_width + hgap)
        ax = fig.add_axes([left, top_bottom, panel_width, row_height])
        graph_plot(
            g,
            nodes_color=no_dict_color,
            ax=ax,
            edge_color_map=edge_color_map,
        )
        no_dict_axes.append(ax)

    # bottom row: 3 "Dict" graphs
    for i, (th, g) in enumerate(zip(thresholds, dict_graphs)):
        left = left_margin + i * (panel_width + hgap)
        ax = fig.add_axes([left, bottom_bottom, panel_width, row_height])
        graph_plot(
            g,
            nodes_color=dict_color,
            ax=ax,
            edge_color_map=edge_color_map,
        )
        dict_axes.append(ax)

    # --- dashed boxes around each graph (no connectors) ----------------------
    # We draw rectangles in figure coordinates around each axes.
    # Box color = corresponding threshold line color
    pad = 0.008  # small padding around each axes

    for i, color in enumerate(line_colors):
        # top row (No Dict.)
        if i < len(no_dict_axes):
            ax = no_dict_axes[i]
            pos = ax.get_position()  # Bbox in figure coordinates
            rect = Rectangle(
                (pos.x0 - pad, pos.y0 - pad),
                pos.width + 2 * pad,
                pos.height + 2 * pad,
                fill=False,
                linestyle='--',
                linewidth=1.5,
                edgecolor=color,
                transform=fig.transFigure,
                clip_on=False,
            )
            fig.add_artist(rect)

        # bottom row (Dict.)
        if i < len(dict_axes):
            ax = dict_axes[i]
            pos = ax.get_position()
            rect = Rectangle(
                (pos.x0 - pad, pos.y0 - pad),
                pos.width + 2 * pad,
                pos.height + 2 * pad,
                fill=False,
                linestyle='--',
                linewidth=1.5,
                edgecolor=color,
                transform=fig.transFigure,
                clip_on=False,
            )
            fig.add_artist(rect)

    for i, (th, color) in enumerate(zip(thresholds, line_colors)):
        # TOP ROW (No Dict.) → label ABOVE
        if i < len(no_dict_axes):
            ax = no_dict_axes[i]
            pos = ax.get_position()
            cx = pos.x0 + pos.width / 2  # center X
            y = pos.y0 + pos.height + 0.01  # slightly above box

            fig.text(
                cx,
                y,
                f'{th:.2f}',
                ha='center',
                va='bottom',
                fontsize=base_size,
                color=color,
                fontweight='bold',
            )

        # BOTTOM ROW (Dict.) → label BELOW
        if i < len(dict_axes):
            ax = dict_axes[i]
            pos = ax.get_position()
            cx = pos.x0 + pos.width / 2  # center X
            y = pos.y0 - 0.015  # slightly below box

            fig.text(
                cx,
                y,
                f'{th:.2f}',
                ha='center',
                va='top',
                fontsize=base_size,
                color=color,
                fontweight='bold',
            )

    # --- Two aligned legends that look like one block ----------------------

    # CASE legend (left "column")
    case_handles = [
        Patch(
            facecolor=case_fill_map['No Dict.'],
            edgecolor='none',
            label='No Dict.',
        ),
        Patch(
            facecolor=case_fill_map['Dict.'],
            edgecolor='none',
            label='Dict.',
        ),
    ]

    leg_case = fig.legend(
        handles=case_handles,
        loc='center right',
        bbox_to_anchor=(0.12, 0.68),  # a bit left
        frameon=False,
        fontsize=base_size,
        title='Method',
        title_fontsize=base_size,
    )

    # EDGE NATURE legend (right "column")
    edge_handles = [
        Line2D(
            [0],
            [0],
            marker='o',
            markersize=14,  # bigger balls in legend only
            linestyle='none',
            color=edge_color_map['Homophilic'],
            label='Homophilic',
        ),
        Line2D(
            [0],
            [0],
            marker='o',
            markersize=14,
            linestyle='none',
            color=edge_color_map['Heterophilic'],
            label='Heterophilic',
        ),
    ]

    leg_edge = fig.legend(
        handles=edge_handles,
        loc='center right',
        bbox_to_anchor=(0.13, 0.33),  # a bit right
        frameon=False,
        fontsize=base_size,
        title='Edge Nature',
        title_fontsize=base_size,
    )

    # make sure both legends stay (second call would otherwise override)
    fig.add_artist(leg_case)
    fig.add_artist(leg_edge)

    return fig


def main():
    folder = (
        Path.cwd().parent / 'results'
        if Path.cwd().name == 'scripts'
        else Path.cwd() / 'results'
    )

    no_dict = pl.read_parquet(folder / 'temp_alignment_metrics_42.parquet')

    with_dict = pl.read_parquet(
        folder
        / 'alignment_metrics_CF_both_both_splitted_dict_reg_sparsity_200_None_1_learnable_sampling_strategy_proto_42.parquet'
    )

    no_dict = homo_hetero(no_dict)
    with_dict = homo_hetero(with_dict)

    df = pl.concat(
        [
            no_dict.select('edge_id', 'alignment_loss')
            .pipe(homo_hetero)
            .with_columns(pl.lit('No Dict.').alias('Case')),
            with_dict.select('edge_id', 'alignment_loss')
            .pipe(homo_hetero)
            .with_columns(pl.lit('Dict.').alias('Case')),
        ]
    ).with_columns(
        pl.format(
            '[{}]', pl.col('edge_id').cast(pl.List(pl.String)).list.join(', ')
        )
    )

    fig = edges_distributions(df)
    fig.savefig(
        'edges_distributions.pdf',
        bbox_inches='tight',  # <-- trims extra whitespace
        pad_inches=0.02,  # small padding; set to 0 for no padding
        dpi=300,
    )
    return None


if __name__ == '__main__':
    main()
