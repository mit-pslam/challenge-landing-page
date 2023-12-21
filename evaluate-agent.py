import argparse

import yaml
from datetime import datetime
import numpy as np

from aia_challenge.agents import SearchAgent, get_subclass
from aia_challenge.eval import evaluate


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent-config", type=str, required=True, help="Configuration file for agent"
    )
    parser.add_argument(
        "--flight-goggles-path",
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
    parser.add_argument(
        "--seed",
        type=int,
        required=False,
        default=0,
        help="Evaluation seed (int). Setting this ensures repeatable episode sequences. Default is 0.",
    )
    parser.add_argument(
        "--video-directory",
        type=str,
        required=False,
        default=None,
        help="Optional path to directory to store episode videos. Default is None.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    with open(args.agent_config) as f:
        config = yaml.load(f, yaml.Loader)

    policy = get_subclass(config["name"], SearchAgent)(config)

    results = evaluate(
        policy,
        args.flight_goggles_path,
        args.base_port,
        args.episodes,
        args.env_config,
        args.seed,
        args.video_directory
    )

    print(results["summary"])
    filename = 'evaluation.' + datetime.now().isoformat() + '.yaml'
    print("Writing results to " + filename)

    # Convert numpy arrays to lists for yaml serialization
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            results[key] = value.tolist()

    # save results to file
    with open(filename, 'w') as outfile:
        yaml.dump(results, outfile, default_flow_style=False)
