import re
import polars as pl
import pandas as pd
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict


def read_files(setup: str) -> pl.DataFrame:
    folder = (
        Path.cwd().parent / 'results'
        if Path.cwd().name == 'scripts'
        else Path.cwd() / 'results'
    )
    pattern = re.compile(rf'dict_metrics_{setup}_(\d+\.?\d*)_')

    dfs = []

    schemas = {}
    files = sorted(folder.glob('dict_metrics_*_*.parquet'))
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
            print(f'\nColumn: {col}')
            for dtype, fnames in dtype_map.items():
                print(f'  {dtype}:')
                for n in fnames:
                    print(f'    - {n}')

    if not has_conflicts:
        print('No dtype conflicts found across files.')

    for file in folder.glob(f'dict_metrics_{setup}_*.parquet'):
        match = pattern.search(file.stem)
        if match:
            df = pl.read_parquet(file)
            if 'regularizer' in df.columns:
                df = df.with_columns(pl.col('regularizer').cast(pl.Float64))
                dfs.append(df)

    # Concatenate all dataframes into one
    if dfs:
        final_df = pl.concat(dfs, how='vertical')
        print(final_df)
    else:
        print(f'No files found matching the pattern in {folder}.')

    print(final_df.columns)
    print(final_df.filter(pl.col('regularizer') == 100000))
    return final_df


# def regularization_plot(df: pl.DataFrame) -> None:
#     # pdf = dfs.to_pandas()
#     sns.lineplot(data=df, x='regularizer', y='nmse')
#     plt.show()

#     sns.lineplot(data=df, x='regularizer', y='sparsity', hue='agent_id')
#     plt.show()
#     return None


def regularization_plot(df: pl.DataFrame) -> None:
    pdf = df.to_pandas()

    # Ensure numeric sorting on the x-axis if 'regularizer' is numeric
    if not str(pdf['regularizer'].dtype).startswith(('float', 'int')):
        pdf['regularizer'] = pd.to_numeric(pdf['regularizer'], errors='ignore')

    # xticks = sorted(pdf['regularizer'].unique())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=False)

    # NMSE vs regularizer
    sns.lineplot(data=pdf, x='regularizer', y='nmse', ax=axes[0])
    axes[0].set_title('NMSE vs Regularizer')
    axes[0].set_xlabel('Regularizer')
    axes[0].set_ylabel('NMSE')
    axes[0].grid(True, axis='y')

    # Sparsity vs regularizer, colored by agent
    sns.lineplot(
        data=pdf, x='regularizer', y='sparsity', hue='agent_id', ax=axes[1]
    )
    axes[1].set_title('Sparsity vs Regularizer')
    axes[1].set_xlabel('Regularizer')
    axes[1].set_ylabel('Sparsity')
    axes[1].grid(True, axis='y')
    axes[1].legend(title='Agent', loc='best')

    plots_dir = (
        Path.cwd().parent / 'plot'
        if Path.cwd().name == 'scripts'
        else Path.cwd() / 'plot'
    )
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_path = plots_dir / 'regularization_plots.png'

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    print(f'Saved figure to: {out_path}')
    return None


def main():
    setup = 'CF_both_both_splitted'
    df = read_files(setup)
    regularization_plot(df)


if __name__ == '__main__':
    main()
