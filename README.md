# Understanding over-squashing and bottlenecks on graphs via curvature
Code accompanying our paper ([arXiv](https://arxiv.org/abs/2111.14522)) at ICLR 2022, recipient of an Outstanding Paper Honorable Mention.

<p align="center">
  <img width="600" align="center" alt="Manifolds and graphs can both exhibit curvature" src="https://user-images.githubusercontent.com/34721006/167724021-588b9903-795e-4de8-a4ca-88d2ce7e514f.png">
</p>

## Setup
The script setup_env.sh assumes that you have conda installed in ~/miniconda3 (as would result from an installation like [this one](https://waylonwalker.com/install-miniconda/)). If you have it installed elsewhere please change the conda activation step in the first line of setup_env.sh as needed before running
```
source setup_env.sh
```
after which you'll have a conda environment `env_curvature` (name can be changed in .sh file) that you can activate in future. `gdl` is a helper library that we used for our experiments. You also may need to change the cudatoolkit/torch versions depending on your cuda version (we used 11.1).

## Usage
To calculate Balanced Forman curvature, look at
```
from gdl.curvature.numba import balanced_forman_curvature     # if you're on CPU
from gdl.curvature.cuda import balanced_forman_curvature      # if you're on GPU with CUDA
```
To get a torch_geometric dataset preprocessed with SDRF, look at
```
from gdl.data import SDRFDataset
```
and if you're using CUDA set
```
import os
os.environ["DEVICE"] = "cuda"
```
or otherwise set the environment variable.
