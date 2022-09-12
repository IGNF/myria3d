FROM nvidia/cuda:10.2-devel-ubuntu18.04
# An nvidia image seems to be necessary for torch-points-kernel. 
# Also, a "devel" image seems required for the same library

# set the IGN proxy, otherwise apt-get and other applications don't work
# Should be commented out outside of IGN
ENV http_proxy 'http://192.168.4.9:3128/'
ENV https_proxy 'http://192.168.4.9:3128/'

# set the timezone, otherwise it asks for it... and freezes
ENV TZ=Europe/Paris
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Needed to use apt-get afterwards due to CUDA changes described here since April 27, 2022:
# https://forums.developer.nvidia.com/t/notice-cuda-linux-repository-key-rotation/212772
# Not the recommpended method, but else we need wget installed afterwards.
# We changed to 10.2-devel-ubuntu18.04 so that might not be needed. 
RUN apt-get update && apt-get upgrade -y && apt-get install -y wget
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

# all the apt-get installs
RUN apt-get update && apt-get upgrade -y && apt-get install -y \
        software-properties-common  \
        wget                        \
        git                         \
        libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6   # package needed for anaconda

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh \
        && /bin/bash ~/miniconda.sh -b -p /opt/conda \
        && rm ~/miniconda.sh

ENV PATH /opt/conda/bin:$PATH

# Only copy necessary files to set up the environment, 
# to use docker caching if requirements files were not updated.
WORKDIR /setup_env
COPY ./setup_env/ .

# Build the environment
RUN ./setup_env/setup_env.sh

# Copy the repository content in /myria3d 
WORKDIR /myria3d
COPY . .

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "myria3d", "/bin/bash", "-c"]

# the entrypoint garanty that all command will be runned in the conda environment
ENTRYPOINT ["conda",  \   
        "run", \
        "-n", \
        "myria3d"]

