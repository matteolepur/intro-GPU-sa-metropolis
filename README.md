# Introduction to GPU computing with Simulating Annealing and Metropolis-Hastings algorithm

## Getting started
`cd` into the repo directory, then using conda run the following command:

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
