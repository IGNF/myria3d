# Installation

## Setting up a virtual environment

### Prerequisites

We use [anaconda](https://www.anaconda.com/products/individual) to manage virtual environments. 
This makes installing pytorch-related libraries way easier than using pure pip installs.

If you want to use a gpu to speed up model training and inference, make sure that the cuda toolkit is installed on your machine.

```bash
sudo apt install nvidia-cuda-toolkit
```

The  [bash](https://github.com/IGNF/lidar-deep-segmentation/setup_env/) directory contains everything you need to setup a compatible pytorch virtual environement.
You can get it by by cloning the entire project with:

```bash
git clone https://github.com/IGNF/lidar-deep-segmentation
cd lidar-deep-segmentation
```
or by manually downloading its content.

### Environment Installation

To install the environment, run:
```bash
source setup_env/setup_env.sh
```

If you do not have CUDA installed, you need comment out cudatoolkit in `setup_env/requirements.yml`

If you do have CUDA, [check you CUDA version](https://varhowto.com/check-cuda-version/) and be sure `cudatoolkit` version in `setup_env/requirements.yml` matches yours.

Finally, activate the created environmnt by running

```bash
conda activate lidar_multiclass
```

## Install source as a package

If you are interested in running inference from anywhere, the easiest way is to install code as a package in a your virtual environment.

Start by activating the virtual environment with

```bash
conda activate lidar_multiclass
```
Then install from a specific branch from github directly. Argument `branch_name` might be `main`, but could also be `prod`, or a specific release.
```
pip install --upgrade https://github.com/IGNF/lidar-deep-segmentation/tarball/{branch_name} 
```

Alternatively, you can install from sources directly in editable mode with
```bash
pip install --editable .
```