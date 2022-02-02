# challenge-landing-page

Reposity to house a challenge description, installation instructions, training scripts, and a submission interface. 

__Outline__
1. [Task Overview](#Task-Overview)
2. [Logistics](#Logistics)
1. [Installation](#Installation-Instructions)
2. [Submission](#Submitting-a-policy)
2. [Getting started with baselines](#Baselines)

## Task Overview

### Description

### Data sources 

### Evaluation 

## Logistics

## Installation Instructions 

Please see the [installation](doc/installation.md) document for detailed instructions on setting up the challenge infrastructure. 

## Submitting a Policy

Policies must implement the [`SearchAgent`](aia_challenge/agents.py#L33) interface:

```python
class SearchAgent:
    def __init__(self, config: Dict[str, Any]) -> None:
        pass

    def act(self, input: Dict[str, np.ndarray]) -> np.ndarray:
        raise NotImplementedError()

    def reset(self) -> None:
        pass
```

The agent takes a user-specified configuration file as input. 
At each episode step, a dictionary of input observations (RGB and depth images and pose)is passed to the `act()` method. `act()` then provides an action, which consists of the 2D vector containing forward velocity and yaw rate. 
`reset()` is called at the start of every episode, and this method provides an opportunity for state-specific information to be initialized. 

To submit an agent, the user creates a configuration file with the template

```yaml
name: <AGENT_NAME>

CUSTOM_ARG_1: CUSTOM_PARAM_1
...
```

Where `AGENT_NAME` is the agent's class name. 

To evaluate the agent, run

```sh
python train-agent.py --config <AGENT_CONFIG> --flight-goggles-path <FLIGHT_GOGGLES_PATH> --base-port <BASE_PORT>
```

The `RandomAgent` [definition](aia_challenge/agents.py#L212) and corresponding [configuration](configs/eval-random-agent.yaml) provide an example of the submission template. 
   
## Baselines 

We provide a baseline that ingests RGB and depth images via convolutional and recurrent networks. The baseline makes use of the [Rllib](https://docs.ray.io/en/master/rllib/) library and is implemented in PyTorch. Please see the baseline [document](doc/baseline.md) for a more complete description and installation instructions. 

# Issues
- [ ] correct version of pytorch gets wiped by a `rl_navigation` install, which breaks cuda compatibility 

