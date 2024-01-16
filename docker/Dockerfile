FROM nvidia/vulkan:1.2.133-450

ARG DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get --allow-insecure-repositories --allow-unauthenticated update 
RUN apt install -y --allow-unauthenticated cmake libeigen3-dev libopencv-dev libzmqpp-dev libyaml-cpp-dev
RUN apt install -y git g++ wget

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
RUN bash ~/miniconda.sh -b -p $HOME/miniconda
RUN /root/miniconda/bin/conda init bash

# Install stuff from conda for challenge. Note that we install an older version for compatibility with cuda 11.4 (provided with the base image)
RUN /root/miniconda/bin/conda create -y --name aia-challenge python=3.7 meson pkgconfig
RUN echo "source /root/miniconda/bin/activate aia-challenge" >~/.bashrc
ENV PATH="/root/miniconda/envs/aia-challenge/bin:${PATH}"

RUN /root/miniconda/bin/conda run -n aia-challenge conda install -y pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge

# Now, install the challenge dependencies
RUN /root/miniconda/bin/conda run -n aia-challenge pip install \
    git+https://git@github.com/mit-pslam/rl_navigation.git@develop#egg=rl_navigation[rllib] \
    git+https://git@github.com/mit-pslam/rllib-policies.git@master#egg=rllib_policies

# Install the local instance of the challenge code into image.
# Note local changes are included in the image.
COPY . /challenge/challenge-landing-page

# NOTE: By default this Docker image will be configured to evaluate the random agent.
# You should ensure that you have modified local code such that your agent is part of the image and is being evaluated.
COPY configs/eval-random-agent.yaml /challenge/challenge-landing-page/configs/eval-submission-agent.yaml

WORKDIR /challenge/challenge-landing-page
