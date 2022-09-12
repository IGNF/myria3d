# WARNING: you may need to adapt cudatoolkit version to your needs.
# Or remove it altogether in case you do not need CUDA support at all

# Install mamba to create the environment faster
conda install -y mamba -n base -c conda-forge
# Build it
mamba env create -f setup_env/requirements.yml -f
# activate it
conda activate myria3d_pyg2_1_0

mkdir -p ./wheels/
cd ./wheels/
wget https://data.pyg.org/whl/torch-1.11.0%2Bcu113/torch_scatter-2.0.9-cp39-cp39-linux_x86_64.whl
wget https://data.pyg.org/whl/torch-1.11.0%2Bcu113/torch_cluster-1.6.0-cp39-cp39-linux_x86_64.whl
wget https://data.pyg.org/whl/torch-1.11.0%2Bcu113/torch_sparse-0.6.14-cp39-cp39-linux_x86_64.whl
wget https://data.pyg.org/whl/torch-1.11.0%2Bcu113/torch_spline_conv-1.2.1-cp39-cp39-linux_x86_64.whl

pip install torch_scatter-2.0.9-cp39-cp39-linux_x86_64.whl
pip install torch_cluster-1.6.0-cp39-cp39-linux_x86_64.whl
pip install torch_sparse-0.6.14-cp39-cp39-linux_x86_64.whl
pip install torch_spline_conv-1.2.1-cp39-cp39-linux_x86_64.whl

# pip install git+https://github.com/pyg-team/pytorch_geometric.git@2.1.0
pip install git+https://github.com/pyg-team/pytorch_geometric.git

# go back
cd ..