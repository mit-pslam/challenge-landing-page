import time
from collections import namedtuple
from enum import Enum
from typing import Any, Dict, List, Union, Optional

import tqdm
import yaml
from rl_navigation.rllib.search import SearchWrapperEnv
from rl_navigation.tasks.search import SearchEnv

from gym import wrappers as gym_wrappers
import random
import os

from .agents import SearchAgent

EpisodeResult = namedtuple("EpisodeInfo", ["output", "steps", "time"])


DEFAULT_CONFIG = {
    "renderer": "flight_goggles",
    "fields": ["depth", "grayscale"],
    "max_steps": 200,
    "action_mapper": "dubins-car",
    "simulator": "dubins-car",
    "flight_goggles_scene": "ground_floor_car",
    "max_range_from_ownship": 20,
    "success_dist": 3,
    "enforce_target_in_fov": True,
    "camWidth": 256,
    "camHeight": 192,
}


class EpisodeOutcome(Enum):
    """Enum defining possible search outcomes."""

    FOUND_TARGET = 0
    COLLIDED = 1
    TIME_EXPIRED = 2

    @staticmethod
    def from_conditions(reached_goal: bool, collided: bool):
        if reached_goal:
            return EpisodeOutcome.FOUND_TARGET
        elif collided:
            return EpisodeOutcome.COLLIDED
        else:
            return EpisodeOutcome.TIME_EXPIRED


def evaluate_episode(env: SearchEnv, policy: SearchAgent) -> EpisodeResult:
    """Run `policy` in `env` for one episode, then return the
    episode results

    Parameters
    ----------
    env: SearchEnv
        OpenAI Gym environment that implements the challenge
        target search task.
    policy: SearchAgent
        Policy that implements the `SearchAgent` interface.

    Returns
    -------
    EpisodeResult
        Namedtuple describing the outcome (found target, collided, or ran out of
        steps), number of steps taken, and duration (in seconds) of the episode.
    """
    start_time = time.time()
    obs = env.reset()
    policy.reset()

    for i in tqdm.tqdm(range(env.max_steps), leave=False):
        action = policy.act(obs)
        obs, reward, done, info = env.step(action)

        if done:
            break

    time_elapsed = time.time() - start_time

    return EpisodeResult(
        EpisodeOutcome.from_conditions(info["reached_goal"], info["collided"]),
        i,
        time_elapsed,
    )


def get_task_config(
    flight_goggles_path: str, base_port: int, custom_config: Union[str, None] = None
) -> Dict[str, Any]:
    """Get configuration for the `SearchWrapperEnv` env,
    which implements the challenge target search task.

    Parameters
    ----------
    flight_goggles_path: str
        Path to flightgoggles executable.
    base_port: int
        Base port for flightgoggles to use.
    custom_config: str
        Path to yaml file of custom configurations.

    Returns
    -------
    Dict[str, Any]
        Target search task configuration.
    """
    config = DEFAULT_CONFIG.copy()
    config["flight_goggles_path"] = flight_goggles_path
    config["base_port"] = base_port

    if custom_config is not None:
        with open(custom_config) as f:
            updated_config = yaml.load(f, yaml.Loader)
        config.update(updated_config)
    return config


def evaluate(
    policy: SearchAgent,
    flight_goggles_path: str,
    base_port: int,
    n_episodes: int,
    custom_task_config: str,
    seed: Optional[int] = None,
    video_directory: os.PathLike = None,
) -> List[EpisodeResult]:
    """Run policy evaluation in target search task.

    The policy is defined by `policy` and the task is defined
    by `SearchWrapperEnv`.

    Parameters
    ----------
    policy: SearchAgent
        A policy the implements the `SearchAgent` inferface.
    flight_goggles_path: str
        Path to flightgoggles executable.
    base_port: int
        Base port for flightgoggles to use.
    n_episodes: int
        Number of episodes to run.
    custom_task_config: str
        Path to yaml file containing any custom task configurations.
    """
    task_config = get_task_config(flight_goggles_path, base_port, custom_task_config)
    env = SearchWrapperEnv(task_config)

    if video_directory is not None:
        env = gym_wrappers.Monitor(
            env=env,
            directory=video_directory,
            video_callable=lambda _: True,
            force=True,
        )

    results = []

    if seed is not None:
        random.seed(seed)
        seeds = [random.randint(0, 2**32) for _ in range(n_episodes)]

    for i in tqdm.tqdm(range(n_episodes)):
        if seed is not None:
            random.seed(seeds[i])
        results.append(evaluate_episode(env, policy))
    env.close()
    return results
