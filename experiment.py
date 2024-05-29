import time

from itertools import product

from src.experiment import SpeedExperiment
from src.scheduler import ConstantBetaScheduler
from src.benchmark_utils import create_benchmarks, plot_benchmarks


def main():
    # run experiments with all possible combination of the following settings
    settings_machine = [True, False]
    settings_batch = [None, 5]
    settings_num_repeats = [10, 100]
    settings_dim = [10, 100]

    # perform experiment
    run_machine = []
    run_batch = []
    run_num_repeats = []
    run_dim = []
    run_total_time = []
    for m, b, r, dim in product(settings_machine, settings_batch, settings_num_repeats, settings_dim):
        experiment = SpeedExperiment(M=100, N=dim, t_max=100,
                                     beta_scheduler=ConstantBetaScheduler(0.5), algorithm="Metropolis",
                                     batch_size=b,
                                     use_gpu=m)

        # run experiment and tabulate execution time
        start = time.time()
        for i in range(r):
            _ = experiment.run()
        end = time.time()
        run_time = end - start

        # store experiment results
        run_machine.append(m)
        run_batch.append(b)
        run_num_repeats.append(r)
        run_dim.append(dim)
        run_total_time.append(run_time)

    # plot results
    df_benchmarks = create_benchmarks(run_machine, run_batch, run_num_repeats, run_dim, run_total_time)
    plot_benchmarks(df_benchmarks)


if __name__ == "__main__":
    main()

