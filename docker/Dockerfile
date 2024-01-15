FROM nvidia/vulkan:1.2.133-450

ARG DEBIAN_FRONTEND=noninteractive
ARG GITHUB_TOKEN

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

# Now, install the challenge code and dependencies
# Because we're trying to use GITHUB_TOKEN, we can't quite do a "pip install -r aia_challenge/requirements.txt"
# So, we'll install the requirements one by one
# Note that for some reason we have to pin setuptools and wheel to specific versions first. Otherwise, we get an error when installing gym
RUN /root/miniconda/bin/conda run -n aia-challenge pip install "setuptools==65.5.0" "wheel==0.38.4"
RUN /root/miniconda/bin/conda run -n aia-challenge pip install "numpy<1.20" "gym<=0.21.0" "importlib-metadata<5.0" "scipy>=1.4.0" "zmq" "opencv-python==4.1.2.30" "transforms3d" "yacs" "matplotlib"
RUN /root/miniconda/bin/conda run -n aia-challenge pip install tesse-gym@git+https://git@github.com/MIT-TESSE/tesse-gym.git@master#egg=tesse-gym
RUN /root/miniconda/bin/conda run -n aia-challenge pip install git+https://${GITHUB_TOKEN}@github.mit.edu/aiia-suas-disaster-response/pyFlightGoggles.git@opencv4-compatible#egg=flightgoggles
RUN /root/miniconda/bin/conda run -n aia-challenge pip install "ray[default,rllib]==1.13.0" "aiohttp<3.8.0" "aioredis==1.3.1"
RUN /root/miniconda/bin/conda run -n aia-challenge pip install --no-deps git+https://${GITHUB_TOKEN}@github.mit.edu/aiia-suas-disaster-response/rl_navigation.git@develop#egg=rl_navigation

RUN /root/miniconda/bin/conda run -n aia-challenge pip install git+https://${GITHUB_TOKEN}@github.mit.edu/aiia-suas-disaster-response/rllib-policies.git@master#egg=rllib_policies
# For some reason, we need to re-install importlib-metadata to get a proper version
RUN /root/miniconda/bin/conda run -n aia-challenge pip install "importlib-metadata<5.0"

# Install the local instance of the challenge code into image.
# Note local changes are included in the image.
COPY . /challenge/challenge-landing-page

# NOTE: By default this Docker image will be configured to evaluate the random agent.
# You should ensure that you have modified local code such that your agent is part of the image and is being evaluated.
COPY configs/eval-random-agent.yaml /challenge/challenge-landing-page/configs/eval-submission-agent.yaml

WORKDIR /challenge/challenge-landing-page
