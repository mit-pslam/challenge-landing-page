# Installation Instructions 

**Notes**

- The following instructions use Anaconda, but other virtual environments, such as [venv](https://docs.python.org/3/library/venv.html), could suffice. 
 - We use several custom repositories (`rl_navigation`, `rllib-policies`). The the following instructions, we install clones of these repositories to enable local development. If you'd prefer not to clone these repositories, you can replace (`git clone git@github.mit.edu:REPO.git && pip install -e REPO` to `pip install git+https://github.mit.edu/REPO.git`



## Setup Python environment


Create a virtual environment (optional):
```sh
conda create --name aia-challenge python=3.7 meson pkgconfig
conda activate aia-challenge
```

Install a few python dependencies:

To use CUDA with PyTorch, follow the Pytorch installation [instructions](https://pytorch.org/) to get the correct version for you system. Simply running `pip install torch`, as listed in the instructions, will install a system-agnostic CPU-only PyTorch build.

```
pip install torch
pip install 'opencv-python-headless<=4.0'
```

## Download flightgoggles

Download binary:


## Clone challenge repo

```sh
git clone git@github.mit.edu:aiia-suas-disaster-response/challenge-landing-page.git
```

## Install the FlightGoggles OpenAI Gym environment

Dependencies
```sh
sudo apt install libeigen3-dev libzmqpp-dev

# Optional, install if you anticipate training across multiple GPUS 
# sudo apt install libvulkan-dev vulkan-validationlayers-dev 
```

Install the `rl_navigation` repo. 

```sh
git clone git@github.mit.edu:aiia-suas-disaster-response/rl_navigation.git
pip install -e rl_navigation[rllib]
```

**Note** Something in the `rl_navigation` install reinstalls pytorch and may disablie CUDA compatibility. If this happens, uninstall and reinstall pytorch (until the issue is fixed) .


## Test Installation

Run an agent that will randomly take actions sampled from a normal distribution 

```
cd challenge-landing-page
python evaluate-agent.py --agent-config configs/eval-random-agent.yaml --episodes 1 --flight-goggles-path <FLIGHT_GOGGLES_PATH> --base-port <BASE_PORT>
```
