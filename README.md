# challenge-landing-page

Free food alert! Someone has generously left pizza(?) in MIT's Stata Center. But there's a catch - food is limited. 
To get the some before it runs out, you will create an autonomous robot that will search the Stata Center and locate this food.

This challenge uses [FlightGoggole's Stata Center environment](https://flightgoggles.mit.edu/virtual-environments/stata-center).
We also provide an OpenAI Gym interface and baseline policies.
This page provides challenges details including a description, installation instructions, a description of the submission interface, and policy training example. 

__Outline__
1. [Task Overview](#Task-Overview)
1. [Installation](#Installation-Instructions)
2. [Submission](#Submitting-a-policy)
2. [Getting started with baselines](#Baselines)

## Task Overview

The agent must find a target placed in the FlightGoggles Stata center environment. 
The target is currently a gate, like the one used for the AlphaPilot challenge, but it will be replaced with a food item at some future date.
For an episode to be considered successful, the agent must find the target within 600 steps. 
The target is considered found if it is within 2 meters of the agent and within the agent's field-of-view. 
Finally, the episode ends if the agent collides with an obstacle.

At each step, the agent observes an RGB image, a ground truth depth image, and ground truth pose.
The agent output a Dubins vehicle-like [action](https://flightgoggles-documentation.scrollhelp.site/fg/Car-Dynamics.374996993.html) via a 2D vector consisting of forward velocity and yaw rate.


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
python evaluate-agent.py --agent-config <AGENT_CONFIG> --flight-goggles-path <FLIGHT_GOGGLES_PATH> --base-port <BASE_PORT>
```

The `RandomAgent` [definition](aia_challenge/agents.py#L212) and corresponding [configuration](configs/eval-random-agent.yaml) provide an example of the submission template. 
Note that `BASE_PORT` is a number that specifies a port for `python` and `FlightGoggles` to communicate with each other.
Port numbers are between 1 to 65535, and typically set to something above 1024 (e.g., 8000).
See [here](https://www.linuxandubuntu.com/home/what-are-ports-how-to-find-open-ports-in-linux) for a bit more details about ports.
   
## Baselines 

We provide a baseline that ingests RGB and depth images via convolutional and recurrent networks. The baseline makes use of the [Rllib](https://docs.ray.io/en/master/rllib/) library and is implemented in PyTorch. 

For complete instructions on running and evaluation our baseline, please see the baseline [document](doc/baseline.md).

## Common Questions/Issues

See our [FAQ](doc/faq.md) page for discussion about common issues that you may experience.

# Issues
- [ ] correct version of pytorch gets wiped by a `rl_navigation` install, which breaks cuda compatibility 

