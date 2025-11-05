import matplotlib.pyplot as plt
from wandb import Image
import seaborn as sns
import pandas as pd


def threshold_study(
    run,
    data: dict,
) -> None:
    df = pd.DataFrame(data)
    fig, ax1 = plt.subplots(figsize=(8, 5))

    sns.lineplot(
        data=df,
        x='threshold',
        y='n_edges',
        ax=ax1,
        color='tab:blue',
        label='Number of edges',
        estimator='mean',
        errorbar=None,
        legend=False,
    )
    ax1.set_xlabel('Edge Loss Threshold', fontsize=16)
    ax1.set_xscale('log')
    ax1.set_ylabel('Number of edges', fontsize=16, color='tab:blue')
    ax2 = ax1.twinx()
    sns.lineplot(
        data=df,
        x='threshold',
        y='task_accuracy',
        ax=ax2,
        color='tab:red',
        label='Accuracy',
        estimator='mean',
        errorbar='sd',
        legend=False,
    )
    ax2.set_ylabel('Network Task accuracy', fontsize=16, color='tab:red')
    sns.despine(right=False)
    # plt.title('Threshold study: accuracy vs number of edges')
    plt.tight_layout()
    run.log({'threshold_study': Image(fig)})
    plt.savefig('threshold_study.pdf')
    plt.close()
    return None
