import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

def create_benchmarks(
        machine: list,
        batch: list,
        num_experiment_executions: list,
        x_dim: list,
        total_time: list
):
    # format some values
    f_machine = []
    for m in machine:
        if m is True:
            f_machine.append("GPU")
        else:
            f_machine.append("CPU")

    f_batch = []
    for b in batch:
        if b is None:
            f_batch.append('Full')
        else:
            f_batch.append(b)

    data = {'machine': f_machine,
            'batch': f_batch,
            'num_experiment_executions': num_experiment_executions,
            'dim': x_dim,
            'total_time': total_time}
    df_benchmarks = pd.DataFrame(data)
    return df_benchmarks


def plot_benchmarks(df_benchmarks):
    plot_dir = Path("plots")
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots()
    sns.boxplot(x='machine', y='total_time', hue='num_experiment_executions', data=df_benchmarks, ax=ax)

    plt.savefig(plot_dir.joinpath("benchmarks.png"))
    plt.close()
