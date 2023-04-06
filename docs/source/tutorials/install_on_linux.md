# Install Myria3D on Linux

## Setting up a virtual environment

### Prerequisites

We use [anaconda](https://www.anaconda.com/products/individual) to manage virtual environments. 
This makes installing pytorch-related libraries way easier than using pure pip installs.

We enable CUDA-acceleration in pytorch as part of the defaut virtual environment recipe (see below).

### Environment Installation

To install the environment, follow these instructions:

```bash
# Install mamba to create the environment faster
conda install -y mamba -n base -c conda-forge
# Build it with mamba
mamba env create -f environment.yml
# activate it
conda activate myria3d
```

Nota: if you do have CUDA, [check you CUDA version](https://varhowto.com/check-cuda-version/) and be sure `cudatoolkit` version in `setup_env/requirements.yml` matches yours. Adapt if needed (including sources for torch-geometric's dependencies given as wheels).


Finally, activate the created environment by running

```bash
conda activate myria3d
```

## Install source as a package

If you are interested in running inference from anywhere, the easiest way is to install code as a package in a your virtual environment.

Start by activating the virtual environment with

```bash
conda activate myria3d
```
Then install from a specific branch from github directly. Argument `branch_name` might be `main`, `dev`, or a specific release.
```
pip install --upgrade https://github.com/IGNF/myria3d/tarball/{branch_name} 
```

Alternatively, you can install from sources directly in editable mode with
```bash
pip install --editable .
```

        
## Troubleshooting

- *OSError(libcusparse.so.11 cannot open shared object file no such file or directory)* ([**](https://github.com/pyg-team/pytorch_geometric/issues/2040#issuecomment-766610625))
    - open the .bashrc file from your Ubuntu home directory and at the end of the file, add the following line (replace anaconda3 with miniconda3 if needed)

            export LD_LIBRARY_PATH="/home/${USER}/anaconda3/envs/myria3d/lib:$LD_LIBRARY_PATH" 
