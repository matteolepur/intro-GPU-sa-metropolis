from src.experiment import SpeedExperiment
from src.scheduler import ConstantBetaScheduler
from src.utils import plot_errors_energies


def main():
    experiment = SpeedExperiment(M=10,
                                 N=10,
                                 t_max=100,
                                 beta_scheduler=ConstantBetaScheduler(0.5),
                                 algorithm="Metropolis",
                                 batch_size=None,
                                 use_gpu=False)
    errors, energies, x = experiment.run()
    plot_errors_energies(errors, energies)


if __name__ == "__main__":
    main()

