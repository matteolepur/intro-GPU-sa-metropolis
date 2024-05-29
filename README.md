# Introduction to GPU computing with Simulating Annealing and Metropolis-Hastings algorithm

## Getting started
You will need to download and install conda.
Follow the instructions at this [page](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html).

For those on Sockeye add the following to your `~/.bashrc`:
```{commandline}
module load miniconda3
```

Once conda is installed, `cd` into the repo directory, then run the following command:

```{commandline}
conda env create --file environment.yml
```
This will create a conda enviorment called `intro-gpu-sa-metropolis`.
Activate this envoirment with the following command:
```commandline
conda activate intro-gpu-sa-metropolis
```
You should be set to run the time experiment
```commandline
python experiment.py
```
This will output a single box plot of the execution times for experiments on the GPU versus CPU.

![test](plots/benchmarks.png)


If you look into `experiment.py` and `example.py` we can output more plots and run more experiments easily.

## Goal 
For a fixed matrix $W$ $\in$ $\mathbb{R}^{m \times n}$ and vector $Y$ $\in \mathbb{R}^{m}$, we want infer $x$ $\in$ $\{-1,1\}^{n}$ for the following model:

```math
Y = \frac{1}{n}ReLU(Wx).
```

We will formulate this problem as an optimization problem using [simulated annealing](https://en.wikipedia.org/wiki/Simulated_annealing) that uses a Metropolis step to target the following [Gibbs-Boltzman](https://en.wikipedia.org/wiki/Boltzmann_distribution) distribution:
```math
p_Y(x) = \frac{1}{Z_{\beta}} * \exp(-\beta H_{Y}(x))
```
where 
```math
H_{Y}(x) = \sum_{m}(Y_{m} - \hat{y}_{m})^{2}
```
will be refered to as the energy.
I believe the key here is that the modes of this target distribution will correspond to states with minimum energy.

### Metropolis Algorithm 
Calculating the MH ratio largely simplifies to calculating the difference between the energy of the two states.

```math
\Delta_{ij} = H_{Y}(x_{j}) - H_{Y}(x_{i})
```

```math
p_{accept}
= 
\min \Big(1, e^{\Delta_{ij}} \Big)
```
