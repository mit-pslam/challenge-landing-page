# Installation Instructions 

These instructions assume Ubuntu and an NVidia GPU.
You will need to have CUDA installed, as well.

**Notes**

- The following instructions use Anaconda, but other virtual environments, such as [venv](https://docs.python.org/3/library/venv.html), could suffice. 
- We use several custom repositories (`rl_navigation`, `rllib-policies`).


## Dependencies

**Libraries**

Install the following dependencies.
```sh
sudo apt install libeigen3-dev libzmqpp-dev

# Optional, install the following if you anticipate training across multiple GPUs 
# sudo apt install libvulkan-dev vulkan-validationlayers-dev 
```

**FlightGoggles**

Next, download the FlightGoggles renderer as described in the [FlightGoggles documentation](https://flightgoggles-documentation.scrollhelp.site/fg/FlightGoggles-Renderer.327286792.html).

## Python Setup

**Environment**

Create a virtual environment (optional):
```sh
conda create --name aia-challenge python=3.7 meson pkgconfig
conda activate aia-challenge
```

**PyTorch**

(Note: We recommend PyTorch for training, and provide examples using PyTorch.
However, PyTorch is not required for evaluation.
You can skip this step and still submit solutions for this challenge.)

Check your version of CUDA (i.e., `nvcc --version`) and install the proper version of [pytorch](https://pytorch.org/get-started/locally/) for your system for use with Python 3.7.

To check that `pytorch` can use the GPU, open up Python 3.7 and run the following. Confirm `True` is returned.
```py
>>> import torch
>>> torch.cuda.is_available()
True
```

**Clone**

Next, clone this repository and use `pip` to install it's requirements.
```sh
git clone git@github.mit.edu:aiia-suas-disaster-response/challenge-landing-page.git
pip install -r challenge-landing-page/requirements.txt
```

## Test Installation

Run an agent that will randomly take actions sampled from a normal distribution 

```sh
cd challenge-landing-page
python evaluate-agent.py --agent-config configs/eval-random-agent.yaml --episodes 1 --flight-goggles-path <FLIGHT_GOGGLES_PATH> --base-port <BASE_PORT>
```
