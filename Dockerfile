FROM mambaorg/micromamba:focal-cuda-11.3.1
# focal is Ubuntu 20.04

WORKDIR /app

COPY environment.yml environment.yml

# Switching to root does not seem necessary in the general case, but the github ci/cd process
# does not seem to work without (rresults in a permission error when running pip packages
# installation similar to https://github.com/mamba-org/micromamba-docker/issues/356)
USER root

RUN micromamba env create -f /app/environment.yml

ENV PATH=$PATH:/opt/conda/envs/myria3d/bin/
# Need to export this for torch_geometric to find where cuda is.
# See https://github.com/pyg-team/pytorch_geometric/issues/2040#issuecomment-766610625
ENV LD_LIBRARY_PATH="/opt/conda/envs/myria3d/lib/:$LD_LIBRARY_PATH"

# Check success of environment creation.
RUN python -c "import torch_geometric;"

# use chown to prevent permission issues
COPY . .

# locate proj
ENV PROJ_LIB=/opt/conda/envs/myria3d/share/proj/

# Check that myria3d can run
RUN python run.py task.task_name=predict --help

# # Set the default command to bash for image inspection.
# CMD ["bash"]
