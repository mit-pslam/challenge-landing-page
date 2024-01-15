# challenge-landing-page

Free food alert! Someone has generously left pizza(?) in MIT's Stata Center. But there's a catch - food is limited. 
To get the some before it runs out, you will create an autonomous robot that will search the Stata Center and locate this food.

This challenge uses [FlightGoggle's Stata Center environment](https://flightgoggles.mit.edu/virtual-environments/stata-center).
We also provide an OpenAI Gym interface and baseline policies.
This page provides challenges details including a description, installation instructions, a description of the submission interface, and policy training example. 

__Outline__
1. [Task Overview](#Task-Overview)
1. [Installation](#Installation-Instructions)
1. [Creating your own policy](#Policy-Design-Overview)
1. [Getting started with a baseline](#Baseline)

1. [Submitting your policy for evaluation](#Submitting)
1. [Additional details](#additional-details)


## Task Overview

The agent must find a target placed in the FlightGoggles Stata center environment. 
The target is currently a gate, like the one used for the AlphaPilot challenge, but it will be replaced with a food item at some future date.
For an episode to be considered successful, the agent must find the target within 1,000 steps.
The target is considered found if it is within 2 meters of the agent and within the agent's field-of-view. 
Finally, the episode ends if the agent collides with an obstacle.

At each step, the agent observes a grayscale image, a ground truth depth image, and ground truth pose.
The agent output a Dubins vehicle-like [action](https://flightgoggles-documentation.scrollhelp.site/fg/Car-Dynamics.374996993.html) via a 2D vector consisting of forward velocity and yaw rate.

### Policy Scoring 
Agent policies are evaluated on 100 episodes.
The goal is to maximize the number of targets found over those episodes.
In the event of a tie, the agent policy with the lowest average number of steps to find a target wins.

## Installation Instructions 

Please see the [installation](doc/installation.md) document for detailed instructions on setting up the challenge infrastructure.
You should ensure that you can run the code in the test installation after completing the install.


## Policy Design Overview

Challenge submissions are *policies*, which receive observations in the simulation environment and decide how to act.
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
At each episode step, a dictionary of input observations (RGB and depth images and pose) is passed to the `act()` method. `act()` then provides an action, which consists of the 2D vector containing forward velocity and yaw rate. 
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


## Baseline

We provide a baseline that ingests RGB and depth images via convolutional and recurrent networks.
The baseline makes use of the [Rllib](https://docs.ray.io/en/master/rllib/) library to perform reinforcement learning and is implemented in PyTorch.

For complete instructions on running and evaluating our baseline, please see the baseline [document](doc/baseline.md).


# Submission

A policy is submitted as a Docker image.
The docker image will contain Python code for the challenge, which includes this repository.
You will need to ensure that your policy is defined in that image, as well.

Details for building, testing, and modifying the challenge Docker image can be found in the [docker overview](doc/docker.md).

## Additional Details

See our [FAQ](doc/faq.md) page for discussion about common issues that you may experience.


## Acknowledgement
Research was sponsored by the United States Air Force Research Laboratory and the Department of the Air Force Artificial Intelligence Accelerator and was accomplished under Cooperative Agreement Number FA8750-19-2-1000.
The views and conclusions contained in this document are those of the authors and should not be interpreted as representing the official policies, either expressed or implied, of the Department of the Air Force or the U.S. Government.
The U.S. Government is authorized to reproduce and distribute reprints for Government purposes notwithstanding any copyright notation herein.
