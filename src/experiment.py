import torch
import math
from . import utils
from . import algorithms


class SpeedExperiment(object):
    """
    If batch_size is set to None, the standard algorithm will be applied.
    When a batch_size is provided, the MCMC algorithm is performed parallelly on multiple batches.
    """
    def __init__(self, N, M, t_max, beta_scheduler, algorithm="Metropolis", batch_size=None, use_gpu=False, W=None, X=None, Y=None):
        self.N = N
        self.M = M
        self.t_max = t_max
        self.batch_size = batch_size

        self.beta_scheduler = beta_scheduler
        self.beta_scheduler.batch_size = batch_size

        if algorithm == "Metropolis":
            self.chain = algorithms.Metropolis()
        else:
            raise Exception("Unknown algorithm.")

        self.use_gpu = use_gpu

        if None not in (W, X, Y):
            # Use the parameters W, X, and Y in the experiment when provided by the user.
            self.W = W
            self.X = X
            self.Y = Y
        else:
            self.draw_parameters()

    def draw_parameters(self):
        print("Drawing random parameters...")
        self.W = torch.randn(self.M, self.N)
        self.X = (2 * torch.randint(2, (self.N,)) - 1).float()
        self.Y = torch.matmul(self.W, self.X).relu() / math.sqrt(self.N)

        if self.use_gpu:
            if torch.cuda.is_available():
                # Transfer to the GPU
                print("Performing MCMC computations on GPU.")
                self.W, self.X, self.Y = self.W.cuda(), self.X.cuda(), self.Y.cuda()
            else:
                print("GPU or CUDA is not available. Performing MCMC computations on CPU by default.")
        else:
            print("Performing MCMC computations on CPU.")

    def run(self):
        if self.batch_size is not None:
            return self.run_batch()

        print("Running experiment in standard mode...")

        # Initialize errors and energies
        errors = self.W.new_tensor([])
        energies = self.W.new_tensor([])

        # Initialize the x vector
        x = (2 * self.W.new(self.N).random_(2) - 1).float()

        # The random values which will be compared to the acceptance probabilities
        rands = self.W.new(self.t_max).uniform_()

        # The indices which will be flipped in x at each iteration
        idxs = self.W.new(self.t_max).random_(self.N).long()

        # Compute the initial value of the energy
        energy, wx = utils.compute_energy(x, self.W, self.Y)

        for iteration in range(self.t_max):
            # Update the value of beta according to the cooling strategy
            self.beta_scheduler.step(energies)

            x, energy, wx = self.chain.step(x, self.W, self.Y, self.beta_scheduler.beta, energy, wx, idxs[iteration], rands[iteration])
            energies = torch.cat((energies, energy.unsqueeze(0)))

            # Compute the current reconstruction error
            e = utils.compute_reconstruction_error(x, self.X)

            errors = torch.cat((errors, e.unsqueeze(0)))

        return errors, energies, x

    def run_batch(self):
        """
        Batch-wise version of run.
        """

        print("Running experiment with batch_size {}...".format(self.batch_size))

        errors = self.W.new_tensor([])
        energies = self.W.new_tensor([])

        # Initialize the x vector
        x = (2 * self.W.new(self.N, self.batch_size).random_(2) - 1).float()

        # The random values which will be compared to the acceptance probabilities
        rands = self.W.new(self.t_max, self.batch_size).uniform_()

        # The indices which will be flipped in x at each iteration
        idxs = self.W.new(self.t_max, self.batch_size).random_(self.N).long()

        # Compute the initial value of the energy
        energy, wx = utils.compute_energy_batch(x, self.W, self.Y)

        for iteration in range(self.t_max):
            # Update the value of beta according to the cooling strategy
            self.beta_scheduler.step(energies)

            x, energy, wx = self.chain.step_batch(x, self.W, self.Y, self.beta_scheduler.beta, energy, wx, idxs[iteration], rands[iteration])
            energies = torch.cat((energies, energy.unsqueeze(0)))

            # Compute the current reconstruction error
            e = utils.compute_reconstruction_error_batch(x, self.X)
            errors = torch.cat((errors, e.unsqueeze(0)))

        return errors, energies, x
