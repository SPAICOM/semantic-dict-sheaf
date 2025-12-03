import re
import hydra
import numpy as np
import polars as pl
import pandas as pd
from pathlib import Path
import seaborn as sns
from typing import Optional
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from collections import defaultdict

try:
    base_dir = Path(__file__).parent.parent
except NameError:
    base_dir = Path.cwd()

style_path = base_dir / '.conf' / 'plotting' / 'plt.mplstyle'
plt.style.use(style_path.resolve())


def read_files(study: str) -> pl.DataFrame:
    folder = (
        Path.cwd().parent / 'results'
        if Path.cwd().name == 'scripts'
        else Path.cwd() / 'results'
    )

    if study == 'proto':
        proto_path = folder / 'network_performance_old.parquet'
        if not proto_path.exists():
            raise FileNotFoundError(
                f'Proto results file not found: {proto_path}'
            )
        df = pl.read_parquet(proto_path, low_memory=True)
        return df

    dfs = pl.read_parquet(folder / f'{study}*').fill_null(384)
    if study == 'alignment':
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
        dfs = (
            dfs.sort('alignment_loss', descending=False)
            .group_by('sparsity', maintain_order=True)
            .agg(
                pl.col('alignment_loss'),
                pl.col('edge_id'),
            )
            .explode('alignment_loss', 'edge_id')
            .with_columns(
                pl.when(pl.col('edge_id').is_in(homo_edges))
                .then(pl.lit('homophilic'))
                .otherwise(pl.lit('heterophilic'))
                .alias('nature')
            )
            .with_columns(
                pl.when(pl.col('sparsity').is_null())
                .then(pl.lit('Dictionary'))
                .otherwise(pl.lit('No dictionary'))
                .alias('Case')
            )
            .with_columns(pl.col('sparsity').fill_null(384))
        )
    elif study == 'dict':
        model_map = {
            0: 'vit_small_patch16_224',
            1: 'vit_small_patch16_384',
            2: 'vit_small_patch32_224',
            3: 'vit_small_patch32_384',
            4: 'levit_128',
            5: 'levit_192',
            6: 'efficientvit_m4',
            7: 'levit_conv_128',
            8: 'volo_d1_224',
            9: 'volo_d1_384',
        }

        df_models = pl.DataFrame(
            {
                'agent_id': list(model_map.keys()),
                'model_name': list(model_map.values()),
            }
        )
        dfs_edge = pl.read_parquet(folder / 'alignment*')
        print(dfs_edge.columns)
        dfs_edge = (
            dfs_edge.filter(pl.col('alignment_loss') <= 0.76)
            .group_by('sparsity')
            .len()
            .rename({'len': 'edge_count'})  # optional but helpful
        )

        dfs = dfs.join(dfs_edge, on='sparsity', how='left')
        dfs = dfs.with_columns(pl.col('degree').list.get(1))
        dfs = dfs.join(df_models, on='agent_id', how='left')

    return dfs


def dict_learning_plot(
    df: pl.DataFrame,
    setup: str,
) -> None:
    pdf = df.to_pandas()

    if setup == 'CF_both_both_splitted_dict_reg':
        pdf['lambda'] = pd.to_numeric(pdf['lambda'], errors='ignore')
        label = r'Sparse Regularizer $\lambda$'
        property = 'lambda'
        metric = 'nmse'
        metric_label = 'NMSE'
        second_label = 'Avg. Accuracy'
        second_metric = 'acc'
        hue = None
    elif setup == 'local_pca':
        label = 'Explained Variance'
        property = 'explained_variance'
        metric = 'acc'
        metric_label = 'Avg. Accuracy'
        second_label = 'Dimensionality'
        second_metric = 'sparsity'
        hue = 'agent_id'
    else:
        raise NotImplementedError

    # three plots side by side
    fig, axes = plt.subplots(1, 3, figsize=(30, 10), sharex=False)

    # === 1. metric vs property ===
    sns.lineplot(
        data=pdf,
        x=property,
        y=metric,
        hue=hue,
        style=hue,
        markers=True,
        ax=axes[0],
    )
    axes[0].set_xlabel(label)
    axes[0].set_ylabel(metric_label)
    axes[0].grid(True, axis='y')
    if setup != 'local_pca':
        axes[0].set_xscale('log')
    else:
        sns.move_legend(axes[0], 'best', title='Agent', frameon=True)

    # === 2. second_metric vs property ===
    sns.lineplot(
        data=pdf,
        x=property,
        y=second_metric,
        hue='agent_id' if 'agent_id' in pdf.columns else None,
        style='agent_id' if 'agent_id' in pdf.columns else None,
        markers=True,
        ax=axes[1],
        legend='full',
    )

    axes[1].set_xlabel(label)
    axes[1].set_ylabel(second_label)
    axes[1].grid(True, axis='y')
    if setup != 'local_pca':
        axes[1].set_xscale('log')

    # === 3. sparsity vs property ===
    sns.lineplot(
        data=pdf,
        x=property,
        y='sparsity',
        hue='agent_id' if 'agent_id' in pdf.columns else None,
        style='agent_id' if 'agent_id' in pdf.columns else None,
        markers=True,
        ax=axes[2],
        legend=False,
    )
    axes[2].set_xlabel(label)
    axes[2].set_ylabel('Non-zero coefficients')
    axes[2].grid(True, axis='y')
    if setup != 'local_pca':
        axes[2].set_xscale('log')

    # === unified legend (only if agent_id exists and we got handles) ===
    if 'agent_id' in pdf.columns:
        handles, labels = axes[2].get_legend_handles_labels()
        if handles and labels:
            fig.legend(
                handles,
                labels,
                title='Agent',
                loc='upper center',
                ncol=len(labels),
            )

    plots_dir = (
        Path.cwd().parent / 'plot'
        if Path.cwd().name == 'scripts'
        else Path.cwd() / 'plot'
    )
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_path = plots_dir / 'regularizer_plots.pdf'

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path)
    plt.close(fig)

    print(f'Saved figure to: {out_path}')
    return None


def alignment_plot(df: pl.DataFrame) -> None:
    pdf = df.to_pandas()

    # Ensure hue is scalar/categorical (not an array). Convert arrays/lists → tuple → string.
    if 'edge_id' in pdf.columns:
        pdf['edge_id'] = (
            pdf['edge_id']
            .apply(
                lambda x: tuple(x.tolist())
                if isinstance(x, np.ndarray)
                else (tuple(x) if isinstance(x, list) else x)
            )
            .astype(str)
            .astype('category')
        )

    # (Optional) draw nicer lines: sort by x within each edge
    if {'edge_id', 'explained_variance'}.issubset(pdf.columns):
        pdf = pdf.sort_values(['edge_id', 'explained_variance'])

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(
        data=pdf,
        x='explained_variance',
        y='alignment_loss',
        hue='edge_id',
        style='edge_id',
        linewidth=2.5,  # thicker lines
        markersize=10,
        markers=True,
        ax=ax,
    )
    ax.set_xlabel('Explained Variance')
    ax.set_ylabel('Edge Alignment Loss')
    ax.legend(title='Edge', loc='best')

    plots_dir = (
        Path.cwd().parent / 'plot'
        if Path.cwd().name == 'scripts'
        else Path.cwd() / 'plot'
    )
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_path = plots_dir / 'alignment_plots.png'

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    print(f'Saved figure to: {out_path}')
    return None


def plot_accuracy_and_edge_loss(
    df: pl.DataFrame,
    figsize: tuple[int, int] = (8, 5),
    title: Optional[str] = None,
) -> None:
    """
    Plot accuracy and edge_loss vs n_proto from a Polars DataFrame.

    Seaborn does:
      - mean aggregation over n_proto
      - standard deviation error bars (errorbar='sd')

    Expected columns in `df`:
        - 'n_proto'    : integer
        - 'accuracies' : list/array of floats
        - 'edges_loss' : list/array of floats
    """
    required_cols = {'n_proto', 'accuracies', 'edges_loss'}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f'DataFrame is missing required columns: {missing}')

    # --- Explode with Polars ---
    acc_long = df.explode('accuracies').select(
        'n_proto',
        pl.col('accuracies').alias('accuracy'),
    )

    edge_long = df.explode('edges_loss').select(
        'n_proto',
        pl.col('edges_loss').alias('edge_loss'),
    )

    # Convert to pandas for seaborn
    acc_pd = acc_long.to_pandas()
    edge_pd = edge_long.to_pandas()

    # --- Plotting ---
    sns.set(style='whitegrid')
    fig, ax1 = plt.subplots(figsize=figsize)

    # Left axis: accuracy with std error bars handled by seaborn
    sns.lineplot(
        data=acc_pd,
        x='n_proto',
        y='accuracy',
        marker='o',
        errorbar='sd',  # std-dev error bars
        ax=ax1,
        label='Accuracy',
    )
    ax1.set_xlabel('n_proto')
    ax1.set_xscale('log')
    ax1.set_ylabel('Accuracy')
    ax1.tick_params(axis='y')

    # Right axis: edge_loss (you can also set errorbar='sd' here if you want)
    ax2 = ax1.twinx()
    sns.lineplot(
        data=edge_pd,
        x='n_proto',
        y='edge_loss',
        marker='s',
        errorbar='sd',
        ax=ax2,
        label='Edge loss',
        linestyle='--',
    )
    ax2.set_ylabel('Edge loss')
    ax2.tick_params(axis='y')

    if title is not None:
        fig.suptitle(title)

    # Combine legends from both axes
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc='best')

    fig.tight_layout()
    plt.show()


def acc_sparsity_plot(
    dict_df: pl.DataFrame,
    figsize: tuple[int, int] = (12, 8),
) -> None:
    """
    Plot:
    - sparsity vs accuracy (hue=model_family, style=agent_id) with Seaborn defaults
    - sparsity vs number of edges (secondary Y axis)
    - two legends inside the plot:
        * Model Family (color)
        * Agent Id (style)
    """
    dict_pdf = dict_df.to_pandas()

    required = {'sparsity', 'acc', 'agent_id', 'model_name', 'edge_count'}
    missing = required - set(dict_pdf.columns)
    if missing:
        raise ValueError(f'dict_df missing required columns: {missing}')

    dict_pdf['sparsity'] = pd.to_numeric(dict_pdf['sparsity'], errors='coerce')
    dict_pdf['acc'] = pd.to_numeric(dict_pdf['acc'], errors='coerce')
    dict_pdf['edge_count'] = pd.to_numeric(
        dict_pdf['edge_count'], errors='coerce'
    )
    dict_pdf['agent_id'] = dict_pdf['agent_id'].astype('category')

    # model_family: take prefix before first "_" and capitalize
    dict_pdf['model_family'] = (
        dict_pdf['model_name']
        .astype(str)
        .str.split('_')
        .str[0]
        .str.capitalize()
    )

    dict_pdf = dict_pdf.sort_values(['agent_id', 'sparsity'])

    families = dict_pdf['model_family'].unique().tolist()
    agents = dict_pdf['agent_id'].cat.categories.tolist()
    agents_str = [str(a) for a in agents]  # <-- for matching legend labels

    fig, ax = plt.subplots(figsize=figsize)

    # Main Seaborn plot: let Seaborn decide markers/linestyles
    sns.lineplot(
        data=dict_pdf,
        x='sparsity',
        y='acc',
        hue='model_family',
        # units='agent_id',
        # estimator=None,
        style='model_family',
        markers=True,
        dashes=False,
        palette='tab10',  # custom palette for accuracy colors
        ax=ax,
    )

    ax.set_xlabel(r'Number of selected atoms $(d^\prime)$')
    ax.set_ylabel('Avg. Accuracy')
    ax.grid(True, axis='y')

    # Secondary Y axis: edge count (no markers)
    ax2 = ax.twinx()
    edges_pdf = (
        dict_pdf[['sparsity', 'edge_count']]
        .drop_duplicates()
        .sort_values('sparsity')
    )
    ax2.plot(
        edges_pdf['sparsity'],
        edges_pdf['edge_count'],
        linestyle='--',
        marker='d',
        color='black',
        linewidth=2,
    )
    ax2.set_ylabel('Number of edges')

    # -------- split Seaborn's combined legend --------
    handles, labels = ax.get_legend_handles_labels()

    hue_handles, hue_labels = [], []
    style_handles, style_labels = [], []

    for h, lab in zip(handles, labels):
        if lab in families:
            hue_handles.append(h)
            hue_labels.append(lab)
        elif lab in agents_str:
            style_handles.append(h)
            style_labels.append(lab)

    # Remove original combined legend
    if ax.legend_ is not None:
        ax.legend_.remove()

    # Legend 1: Model Family (colors)
    legend1 = ax.legend(
        hue_handles,
        hue_labels,
        title='Architectural Family',
        loc='center left',
        bbox_to_anchor=(0.55, 0.25),
        framealpha=1.0,
        frameon=True,
        ncol=2,
    )
    ax.add_artist(legend1)

    type_handles = [
        Line2D(
            [0],
            [0],
            color='black',
            linestyle='-',
            marker=None,
            markersize=6,
            label='Agents Accuracy',
        ),
        Line2D(
            [0],
            [0],
            color='black',
            linestyle='--',
            marker=None,
            label='Number of Edges',
        ),
    ]

    legend_types = ax.legend(
        handles=type_handles,
        title='Curve Types',
        loc='center left',
        bbox_to_anchor=(0.61, 0.45),
        frameon=True,
        framealpha=1.0,
    )
    ax.add_artist(legend_types)

    # Save
    plots_dir = (
        Path.cwd().parent / 'plot'
        if Path.cwd().name == 'scripts'
        else Path.cwd() / 'plot'
    )
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_path = plots_dir / 'dict_acc_plots.pdf'

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    print(f'Saved joint sparsity figure to: {out_path}')
    return None


def joint_sparsity_plots(
    align_df: pl.DataFrame,
    dict_df: pl.DataFrame = None,
    figsize: tuple[int, int] = (16, 8),
) -> None:
    """
    Create two plots:

    - Left:  sparsity vs acc      for each agent (from dict_df)
    - Right: sparsity vs alignment_loss for each edge (from align_df)
    """

    if dict_df is not None:
        # --- dict side: sparsity vs acc per agent ---
        dict_pdf = dict_df.to_pandas()

        # ensure numeric sparsity & acc
        if 'sparsity' not in dict_pdf.columns or 'acc' not in dict_pdf.columns:
            raise ValueError(
                "dict_df must contain 'sparsity' and 'acc' columns."
            )

        dict_pdf['sparsity'] = pd.to_numeric(
            dict_pdf['sparsity'], errors='coerce'
        )
        dict_pdf['acc'] = pd.to_numeric(dict_pdf['acc'], errors='coerce')

        if 'agent_id' in dict_pdf.columns:
            dict_pdf['agent_id'] = dict_pdf['agent_id'].astype('category')

        # sort for nicer lines
        sort_cols = (
            ['agent_id', 'sparsity']
            if 'agent_id' in dict_pdf.columns
            else ['sparsity']
        )
        dict_pdf = dict_pdf.sort_values(sort_cols)

    # --- alignment side: sparsity vs alignment_loss per edge ---
    align_pdf = align_df.to_pandas()

    if (
        'sparsity' not in align_pdf.columns
        or 'alignment_loss' not in align_pdf.columns
    ):
        raise ValueError(
            "align_df must contain 'sparsity' and 'alignment_loss' columns."
        )

    align_pdf['sparsity'] = pd.to_numeric(
        align_pdf['sparsity'], errors='coerce'
    )
    align_pdf['alignment_loss'] = pd.to_numeric(
        align_pdf['alignment_loss'], errors='coerce'
    )

    if 'edge_id' in align_pdf.columns:
        align_pdf['edge_id'] = (
            align_pdf['edge_id']
            .apply(
                lambda x: tuple(x.tolist())
                if isinstance(x, np.ndarray)
                else (tuple(x) if isinstance(x, list) else x)
            )
            .astype(str)
            .astype('category')
        )
        align_pdf = align_pdf.sort_values(['edge_id', 'sparsity'])
    else:
        align_pdf = align_pdf.sort_values(['sparsity'])

    # --- plotting ---
    fig, axes = (
        plt.subplots(1, 2, figsize=figsize, sharex=False)
        if dict_df is not None
        else plt.subplots()
    )

    if dict_df is not None:
        ax = axes[1]
        # Left: sparsity vs acc per agent
        sns.lineplot(
            data=dict_pdf,
            x='sparsity',
            y='acc',
            hue='agent_id' if 'agent_id' in dict_pdf.columns else None,
            style='agent_id' if 'agent_id' in dict_pdf.columns else None,
            markers=True,
            ax=axes[0],
        )
        axes[0].set_xlabel('Sparsity')
        axes[0].set_ylabel('Accuracy')
        axes[0].grid(True, axis='y')
        if 'agent_id' in dict_pdf.columns:
            axes[0].legend(title='Agent', loc='best')
        else:
            axes[0].legend_.remove()
    else:
        ax = axes

    # Right: sparsity vs alignment_loss per edge
    sns.lineplot(
        data=align_pdf,
        x='sparsity',
        y='alignment_loss',
        # hue='edge_id' if 'edge_id' in align_pdf.columns else None,
        hue='nature',
        units='edge_id',
        estimator=None,
        # style='edge_id' if 'edge_id' in align_pdf.columns else None,
        markers=True,
        ax=ax,
    )
    ax.set_xlabel('Sparsity')
    # ax.set_yscale('log')
    # ax.set_xscale('log')
    ax.set_ylabel('Edge alignment loss')
    ax.grid(True, axis='y')
    if 'edge_id' in align_pdf.columns:
        ax.legend(title='Edge', loc='best')
    else:
        ax.legend_.remove()

    plots_dir = (
        Path.cwd().parent / 'plot'
        if Path.cwd().name == 'scripts'
        else Path.cwd() / 'plot'
    )
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_path = plots_dir / 'sparsity_joint_plots.pdf'

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    print(f'Saved joint sparsity figure to: {out_path}')
    return None


@hydra.main(
    version_base=None,
    config_path='../.conf/hydra/visualization',
    config_name='local_pca',
)
def main(cfg):
    if cfg.study == 'dict':
        df = read_files('dict')
        dict_learning_plot(df, cfg.setup)
    elif cfg.study == 'alignment':
        df = read_files(cfg.setup, 'alignment')
        alignment_plot(df)
    elif cfg.study == 'proto':
        df = read_files('proto')
        plot_accuracy_and_edge_loss(df)
    elif cfg.study == 'dict_acc':
        dict_df = read_files('dict')
        acc_sparsity_plot(dict_df)
    elif cfg.study == 'dict_alignment':
        # dict_df = read_files('dict')
        dict_df = None
        align_df = read_files('alignment')
        joint_sparsity_plots(align_df, dict_df)
    else:
        raise ValueError(f'Unknown study type: {cfg.study}')


if __name__ == '__main__':
    main()
