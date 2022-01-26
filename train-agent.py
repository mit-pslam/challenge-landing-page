import argparse
import os
from typing import Any, Dict

import ray
import yaml
from ray.rllib.agents import ppo
from ray.rllib.models import ModelCatalog
from ray.tune.logger import pretty_print
from rl_navigation.rllib.search import SearchCallbacks, SearchWrapperEnv
from rl_navigation.rllib.utils import get_logger_creator
from rllib_policies.vision import NatureCNNActorCritic, NatureCNNRNNActorCritic

ModelCatalog.register_custom_model("nature_cnn_rnn", NatureCNNRNNActorCritic)
ModelCatalog.register_custom_model("nature_cnn", NatureCNNActorCritic)


def get_ppo_config(
    config_path: str, flight_goggles_path: str, base_port: int
) -> Dict[str, Any]:
    """Populate PPO config system specific information.

    Parameters
    ----------
    config_path: str
        Path to config of PPO specific settings.
    flight_goggles_path: str
        Path to flightgoggles executable.
    base_port: int
        Base port for flightgoggles to use.

    Returns
    -------
    Dict[str, Any]
        PPO configuration updated with flightgoggles
        pathl base port, and logging callbacks.
    """
    with open(config_path) as f:
        config = yaml.load(f, yaml.Loader)

    config["callbacks"] = SearchCallbacks
    config["env_config"]["base_port"] = base_port
    config["env_config"]["flight_goggles_path"] = flight_goggles_path

    return config


def set_env_vars(display: str, device: int) -> None:
    """Set display, cuda, and vulkan system variables."""
    os.environ["DISPLAY"] = f"{display}"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)

    # vulkan variables
    os.environ["ENABLE_DEVICE_CHOOSER_LAYER"] = "1"
    os.environ["VULKAN_DEVICE_INDEX"] = str(device)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, help="Path to training config", required=True
    )
    parser.add_argument(
        "--flight-goggles-path",
        type=str,
        help="Path to flightgoggles build",
        required=True,
    )

    # TODO(ZR)  add docs
    parser.add_argument(
        "--base-port", type=int, help="Base flightgoggles port", required=True
    )
    parser.add_argument(
        "--display",
    )
    parser.add_argument("--device")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    config = get_ppo_config(args.config, args.flight_goggles_path, args.base_port)
    log_root = config.pop("log_root")
    exp_name = config.pop("experiment_name")
    num_train_iterations = config.pop("num_training_iterations")
    ckpt_save_freq = config.pop("checkpoint_save_frequency")

    set_env_vars(args.display, args.device)

    ray.init()

    trainer = ppo.PPOTrainer(
        env=SearchWrapperEnv,
        config=config,
        logger_creator=get_logger_creator(
            log_root,
            exp_name,
        ),
    )

    for i in range(num_train_iterations):
        result = trainer.train()
        print(pretty_print(result))

        if (i + 1) % ckpt_save_freq == 0:
            checkpoint = trainer.save()
            # save torch models to enable inference without loading entire PPOTrainer
            trainer.export_policy_model(f"{trainer.logdir}/torch-models/model-{i}")
