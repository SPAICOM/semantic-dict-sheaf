import re
import hydra
import numpy as np
import polars as pl
import pandas as pd
from pathlib import Path
import seaborn as sns
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
    pattern = re.compile(rf'{study}_metrics_{setup}_(\d+\.?\d*)_')

    dfs = []

    schemas = {}
    files = sorted(folder.glob(f'{study}_metrics_*_*.parquet'))
    for f in files:
        df = pl.read_parquet(f, low_memory=True)
        schemas[f.name] = dict(zip(df.columns, df.dtypes))

    # Build a map: column -> dtype -> [files]
    col_dtype_files = defaultdict(lambda: defaultdict(list))
    for fname, sch in schemas.items():
        for col, dtype in sch.items():
            col_dtype_files[col][dtype].append(fname)

    # Print only columns that have >1 dtype across files
    has_conflicts = False
    for col, dtype_map in col_dtype_files.items():
        if len(dtype_map) > 1:
            has_conflicts = True
            # print(f'\nColumn: {col}')
            # for dtype, fnames in dtype_map.items():
            #     print(f'  {dtype}:')
            #     for n in fnames:
            #         print(f'    - {n}')

    if not has_conflicts:
        print('No dtype conflicts found across files.')

    for file in folder.glob(f'{study}_metrics_{setup}_*.parquet'):
        match = pattern.search(file.stem)
        if match:
            print(file)
            df = pl.read_parquet(file)
            print(df.columns)
            if 'lambda' in df.columns:
                df = df.with_columns(pl.col('lambda').cast(pl.Float64))
            dfs.append(df)

    # Concatenate all dataframes into one
    if dfs:
        final_df = pl.concat(dfs, how='vertical')
        print(final_df)
    else:
        print(f'No files found matching the pattern in {folder}.')

    # print(final_df.columns)
    return final_df


def dict_learning_plot(
    df: pl.DataFrame,
    setup: str,
) -> None:
    pdf = df.to_pandas()

    if setup != 'local_pca':
        pdf['lambda'] = pd.to_numeric(pdf['lambda'], errors='ignore')
        label = r'Sparse Regularizer $\lambda$'
        property = 'lambda'
        metric = 'nmse'
        metric_label = 'NMSE'
        second_label = 'Avg. Accuracy'
        second_metric = 'acc'
        hue = None
    else:
        label = 'Explained Variance'
        property = 'explained_variance'
        metric = 'acc'
        metric_label = 'Avg. Accuracy'
        second_label = 'Dimensionality'
        second_metric = 'sparsity'
        hue = 'agent_id'

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
    # handles, labels = axes[1].get_legend_handles_labels()

    # # Keep first occurrence (preserves color)
    # by_label = {}
    # for h, l in zip(handles, labels):
    #     if l not in by_label:
    #         by_label[l] = h

    # axes[1].legend(by_label.values(), by_label.keys(), title='Agent')
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
    # ax.set_title('Edge Alignment Loss vs Explained Variance')
    ax.set_xlabel('Explained Variance')
    ax.set_ylabel('Edge Alignment Loss')
    # ax.grid(True, axis='y')
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


@hydra.main(
    version_base=None,
    config_path='../.conf/hydra/visualization',
    config_name='local_pca',
)
def main(cfg):
    df = read_files(cfg.setup, cfg.study)
    if cfg.study == 'dict':
        dict_learning_plot(df, cfg.setup)
    elif cfg.study == 'alignment':
        pass
        alignment_plot(df)
    else:
        raise ValueError(f'Unknown study type: {cfg.study}')


if __name__ == '__main__':
    main()
