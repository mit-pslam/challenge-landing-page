# Baseline

## Description


## Installation

Install rllib-policies

```
git clone git@github.mit.edu:aiia-suas-disaster-response/rllib-policies.git
pip install -e rllib-policies
```

## Train an agent

```sh
cd challenge-landing-page
python train-agent.py --config configs/train-rllib-agent.yaml --flight-goggles-path <FLIGHT_GOGGLES_PATH> --base-port <BASE_PORT>
```

## Evaluate your agent

```
python evaluate-agent.py --config configs/eval-rllib-agent.yaml --episodes 1 --flight-goggles-path <FLIGHT_GOGGLES_PATH> --base-port <BASE_PORT>
```
