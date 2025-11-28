import re
import hydra
import numpy as np
import polars as pl
import pandas as pd
from pathlib import Path
import seaborn as sns
from typing import Optional
import matplotlib.pyplot as plt
from collections import defaultdict

try:
    base_dir = Path(__file__).parent.parent
except NameError:
    base_dir = Path.cwd()

style_path = base_dir / '.conf' / 'plotting' / 'plt.mplstyle'
plt.style.use(style_path.resolve())


def read_files(setup: str, study: str) -> pl.DataFrame:
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

    # pattern = re.compile(rf'{study}_metrics_{setup}_(\d+\.?\d*)_')

    dfs = pl.read_parquet(folder / f'{study}*')
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
                pl.col('alignment_loss').head(20),
                pl.col('edge_id').head(20),
            )
            .explode('alignment_loss', 'edge_id')
            .with_columns(
                pl.when(pl.col('edge_id').is_in(homo_edges))
                .then(pl.lit('homophilic'))
                .otherwise(pl.lit('heterophilic'))
                .alias('nature')
            )
        )
    # dfs = []

    # schemas = {}
    # files = sorted(folder.glob(f'{study}_metrics_*_*.parquet'))
    # for f in files:
    #     df = pl.read_parquet(f, low_memory=True)
    #     schemas[f.name] = dict(zip(df.columns, df.dtypes))

    # # Build a map: column -> dtype -> [files]
    # col_dtype_files = defaultdict(lambda: defaultdict(list))
    # for fname, sch in schemas.items():
    #     for col, dtype in sch.items():
    #         col_dtype_files[col][dtype].append(fname)

    # # Print only columns that have >1 dtype across files
    # has_conflicts = False
    # for col, dtype_map in col_dtype_files.items():
    #     if len(dtype_map) > 1:
    #         has_conflicts = True

    # if not has_conflicts:
    #     print('No dtype conflicts found across files.')

    # print(folder.glob(f'{study}_metrics_{setup}_*.parquet'))
    # for file in folder.glob(f'{study}_metrics_{setup}_*.parquet'):
    #     match = pattern.search(file.stem)
    #     if match:
    #         print(file)
    #         df = pl.read_parquet(file)
    #         print(df.columns)
    #         if 'lambda' in df.columns:
    #             df = df.with_columns(pl.col('lambda').cast(pl.Float64))
    #         dfs.append(df)

    # # Concatenate all dataframes into one
    # if dfs:
    #     final_df = pl.concat(dfs, how='vertical')
    #     print(final_df)
    # else:
    #     print(f'No files found matching the pattern in {folder}.')
    #     final_df = pl.DataFrame()

    print(dfs)

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


# === NEW: joint sparsity plots for dict + alignment ===
def joint_sparsity_plots(
    dict_df: pl.DataFrame,
    align_df: pl.DataFrame,
    figsize: tuple[int, int] = (16, 8),
) -> None:
    """
    Create two plots:

    - Left:  sparsity vs acc      for each agent (from dict_df)
    - Right: sparsity vs alignment_loss for each edge (from align_df)
    """
    # --- dict side: sparsity vs acc per agent ---
    dict_pdf = dict_df.to_pandas()

    # ensure numeric sparsity & acc
    if 'sparsity' not in dict_pdf.columns or 'acc' not in dict_pdf.columns:
        raise ValueError("dict_df must contain 'sparsity' and 'acc' columns.")

    dict_pdf['sparsity'] = pd.to_numeric(dict_pdf['sparsity'], errors='coerce')
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
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=False)

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
        ax=axes[1],
    )
    axes[1].set_xlabel('Sparsity')
    # axes[1].set_yscale('log')
    axes[1].set_ylabel('Edge alignment loss')
    axes[1].grid(True, axis='y')
    if 'edge_id' in align_pdf.columns:
        axes[1].legend(title='Edge', loc='best')
    else:
        axes[1].legend_.remove()

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
    # Note: we now read files *inside* each branch instead of once at the top,
    # so we can reuse read_files for different studies.
    if cfg.study == 'dict':
        df = read_files(cfg.setup, 'dict')
        dict_learning_plot(df, cfg.setup)
    elif cfg.study == 'alignment':
        df = read_files(cfg.setup, 'alignment')
        alignment_plot(df)
    elif cfg.study == 'proto':
        df = read_files(cfg.setup, 'proto')
        plot_accuracy_and_edge_loss(df)
    # === NEW CASE: read both dict_metrics and alignment_metrics ===
    elif cfg.study == 'dict_alignment':
        dict_df = read_files(cfg.setup, 'dict')
        align_df = read_files(cfg.setup, 'alignment')
        print(dict_df)
        print(align_df)
        joint_sparsity_plots(dict_df, align_df)
    else:
        raise ValueError(f'Unknown study type: {cfg.study}')


if __name__ == '__main__':
    main()
