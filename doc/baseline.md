# Baseline

## Description

Our baseline policy is composed of convolutional and recurrent networks, and takes as input RGB and depth image. We train our baseline using Proximal Policy Optimization (PPO) in [RLlib](https://docs.ray.io/en/latest/rllib/index.html).

The baseline is implemented in [`agents.py`]().


## Installation

We implement our baselines in the [rllib-policies](https://github.mit.edu/aiia-suas-disaster-response/rllib-policies) library, which contains utilities for creating RLlib-compatible policies. 
Install this library to run the baseline.

```
git clone git@github.mit.edu:aiia-suas-disaster-response/rllib-policies.git
pip install -e rllib-policies
```

## Train an agent

The `train-agent.py` script provides an example of how to train the baseline using PPO. 
All training-specific arguments are given via a configuration file, like `train-rllib-agent.yaml`. This configuration defines arguments for training, logging, and the baseline network definition.

```sh
cd challenge-landing-page
python train-agent.py --config configs/train-rllib-agent.yaml --flight-goggles-path <FLIGHT_GOGGLES_PATH> --base-port <BASE_PORT>
```

### 
By default, training logs will be written under the `./ray_results` folder. Training logs include sample episode videos and tensorboard logging. Tensorboard may be used to visualize training results in real time. See the Tensorboard [documentation](https://www.tensorflow.org/install) for install instructions. Then, you can run Tensorboard via the following command:

```
>> tensorboard --logdir ray_results
```



## Evaluate your agent

You may evaluate your baseline with the `evaluate-agent.py` script, which implements the [submission Interface](../README.md#submitting-a-policy).
The provided configuration will evaluate a policy defined by `train-rllib-agent.yaml`.

```
python evaluate-agent.py --agent-config configs/eval-rllib-agent.yaml --episodes 1 --flight-goggles-path <FLIGHT_GOGGLES_PATH> --base-port <BASE_PORT>
```
