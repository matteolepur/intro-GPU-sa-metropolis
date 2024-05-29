# Introduction to GPU computing with Simulating Annealing and Metropolis-Hastings algorithm

### Drivers
What are NVIDIA drivers?

### Some useful cli commands when on the cluster
After installing the correct drivers, 
NVIDIA provides a cli program that outputs what GPU your computer has
and statistics about its use.
```commandline
nvidia-smi
```
```commandline
Tue May 28 21:16:50 2024       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.171.04             Driver Version: 535.171.04   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA RTX A2000 Laptop GPU    Off | 00000000:01:00.0 Off |                  N/A |
| N/A   44C    P8               7W /  35W |    788MiB /  4096MiB |     22%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A      1553      G   /usr/lib/xorg/Xorg                           83MiB |
|    0   N/A  N/A      3107      G   /usr/lib/xorg/Xorg                          302MiB |
|    0   N/A  N/A      3281      G   /usr/bin/gnome-shell                        127MiB |
|    0   N/A  N/A      3763      G   ...yOnDemand --variations-seed-version      100MiB |
|    0   N/A  N/A      5031      G   ./jetbrains-toolbox                           7MiB |
|    0   N/A  N/A      8029      G   ...ures=SpareRendererForSitePerProcess       58MiB |
|    0   N/A  N/A      8402      G   ...seed-version=20240528-050051.483000       88MiB |
+---------------------------------------------------------------------------------------+

```
Here we see that my laptop has ...


I found the following command useful especially on the cluster to monitor the GPU usage.
```commandline
watch -n3 nvidia-smi
```
This command runs the `nvidia-smi` command every 3 seconds to provide me with an updated output.

## PyTorch


#### What is PyTorch?
- Jax is ...
- It allows for CPU and GPU usage

#### Using the GPU in PyTorch
- We must declare `.cuda` because ...
- 
```
Tensor.new_tensor(data)
```
Returns a new tensor with the same `dtype` and `device` as Tensor but with the specified data.

```
Tensor.new()
```
Returns a new tensor with the same `dtype` and `device` as Tensor but empty data.

```
Tensor.to()
```
changes the `dtype` and `device` of the Tensor.


## Goal 
For a fixed matrix $W$ $\in$ $\mathbb{R}^{m \times n}$ and vector $Y$ $\in \mathbb{R}^{m}$, we want infer $x$ $\in$ $\{-1,1\}^{n}$ for the following model:
$$
Y = \frac{1}{n}ReLU(Wx).
$$

We will formulate this problem as an optimization problem using [simulated annealing](https://en.wikipedia.org/wiki/Simulated_annealing) that uses a Metropolis step to target the following [Gibbs-Boltzman](https://en.wikipedia.org/wiki/Boltzmann_distribution) distribution:
$$
p_Y(x) = \frac{1}{Z_{\beta}} * \exp(-\beta H_{Y}(x))
$$
where 
$$
H_{Y}(x) = \sum_{m}(Y_{m} - \hat{y}_{m})^{2}
$$
will be refered to as the energy.

### Metropolis Algorithm 
Calculating the MH ratio largely simplifies to calculating the difference between the energy of the two states.

$$
\Delta_{ij} = H_{Y}(x_{j}) - H_{Y}(x_{i})
$$

$$
p_{accept}
= 
\min \Big(1, e^{\Delta_{ij}} \Big)
$$

## Usage 


