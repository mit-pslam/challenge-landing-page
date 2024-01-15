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
        default=None,
        help="Evaluation seed (int). Setting this ensures repeatable episode sequences. Default is None, which requires user input.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=False,
        default="output.yaml",
        help="Path to store results yaml file. Default is output.yaml.",
    )
    parser.add_argument(
        "--video-directory",
        type=str,
        required=False,
        default=None,
        help="Optional path to directory to store episode videos. Default is None.",
    )
    parser.add_argument(
        "--disable",
        action='store_true',
        help="Disable storing of episode seeds and per-episode progress bars. Used for blind evaluations."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    if args.seed is None:
        args.seed = input("Enter seed (int): ")
        print("")  # add line return

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
        args.video_directory,
        args.disable,
    )

    print(results["summary"])
    print("Writing results to " + args.output_file)

    # Convert numpy arrays to lists for yaml serialization
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            results[key] = value.tolist()

    # save results to file
    with open(args.output_file, 'w') as outfile:
        yaml.dump(results, outfile, default_flow_style=False)
