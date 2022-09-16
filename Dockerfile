FROM nvidia/cuda:11.3.1-base-ubuntu20.04

# set the IGN proxy, otherwise apt-get and other applications don't work
# Should be commented out outside of IGN
ENV http_proxy 'http://192.168.4.9:3128/'
ENV https_proxy 'http://192.168.4.9:3128/'


# Remove any third-party apt sources to avoid issues with expiring keys.
RUN rm -f /etc/apt/sources.list.d/*.list

# Install some basic utilities
RUN apt-get update && apt-get install -y \
        curl \
        ca-certificates \
        sudo \
        git \
        bzip2 \
        libx11-6 \
        && rm -rf /var/lib/apt/lists/*


# Set up the Conda environment
ENV CONDA_AUTO_UPDATE_CONDA=false \
        PATH=~/miniconda/bin:$PATH
COPY environment.yml /app/environment.yml
RUN curl -sLo ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-py39_4.10.3-Linux-x86_64.sh \
        && chmod +x ~/miniconda.sh \
        && ~/miniconda.sh -b -p ~/miniconda \
        && rm ~/miniconda.sh \
        && ~/miniconda/bin/conda env update -n base -f /app/environment.yml \
        && rm /app/environment.yml \
        && ~/miniconda/bin/conda clean -ya

# Need to export this for torch_geometric to find where cuda is.
# See https://github.com/pyg-team/pytorch_geometric/issues/2040#issuecomment-766610625
ENV LD_LIBRARY_PATH="~/miniconda/lib/:$LD_LIBRARY_PATH"

# Check succes of environment creation.
RUN python -c "import torch_geometric;"

# Create a working directory
RUN mkdir /app

# Copy the repository content in /app 
WORKDIR /app
COPY . .

# Set the default command to bash for image inspection.
CMD ["bash"]

