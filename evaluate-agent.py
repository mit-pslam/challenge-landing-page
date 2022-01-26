import argparse

import yaml

from aia_challenge.agents import ChallengeSubmission, get_subclass
from aia_challenge.eval import evaluate


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent-config", type=str, required=True, help="Configuration file for agent"
    )
    parser.add_argument(
        "--flightgoggles-path",
        type=str,
        required=True,
        help="Path to flightgoggles executable",
    )
    parser.add_argument(
        "--base-port",
        type=int,
        required=True,
        help="Base port that flightgoggles will use",
    )
    parser.add_argument(
        "--episodes", type=int, default=1, help="Number of episodes to run"
    )
    parser.add_argument(
        "--env-config",
        type=str,
        required=False,
        default=None,
        help="Optional environment configuration",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    with open(args.agent_config) as f:
        config = yaml.load(f, yaml.Loader)

    policy = get_subclass(config["name"], ChallengeSubmission)(config)

    results = evaluate(
        policy, args.flightgoggles_path, args.base_port, args.episodes, args.env_config
    )

    print(results)
