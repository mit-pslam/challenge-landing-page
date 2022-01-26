from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from gym import spaces
from ray.rllib.models.preprocessors import DictFlatteningPreprocessor
from ray.rllib.models.torch.torch_action_dist import TorchDiagGaussian
from ray.rllib.policy import TorchPolicy
from rllib_policies.vision import NatureCNNRNNActorCritic


def get_all_subclasses(cls: object):
    """Get all subclasses of `cls`"""
    return set(cls.__subclasses__()).union(
        [s for c in cls.__subclasses__() for s in get_all_subclasses(c)]
    )


def get_subclass(class_name: str, base_cls: object):
    """Get"""
    sub_classes = [
        sub_class
        for sub_class in get_all_subclasses(base_cls)
        if sub_class.__name__ == class_name
    ]

    if len(sub_classes) == 0:
        raise ValueError(f"{class_name} is not a subclass of {base_cls}")

    if len(sub_classes) > 1:
        raise ValueError(f"Multiple definitions of {class_name} found")

    return sub_classes[0]


class SearchAgent:
    def __init__(self, config_path: str):
        raise NotImplementedError()

    def forward(self, input: Dict[str, np.ndarray]) -> Tuple[np.ndarray, Any]:
        raise NotImplementedError()

    def reset(self) -> None:
        raise NotImplementedError()


class RllibAgent(SearchAgent):
    def __init__(
        self,
        config: str,
    ) -> None:
        """Example submission implementation for the policy
        trained with the `train-agent.py` script.

        This policy is designed for inference only,
        and thus it doesn't require a flightgoggles
        instance, PPO configrations, or any of the other
        more resource-intensive componenets required for
        training.

        Parameters
        ----------
        config_path: str
            Path to policy config file.
        """
        policy_config = self.get_config(config)
        model = NatureCNNRNNActorCritic(
            self.observation_space,
            self.action_space,
            self.action_space.shape[0] * 2,
            model_config={},
            name="Inferer",
            **policy_config["model"]["custom_model_config"],
        )
        model.load_state_dict(torch.load(policy_config["weights"]).state_dict())

        self.preprocessor = DictFlatteningPreprocessor(self.observation_space)
        self.policy = TorchPolicy(
            observation_space=self.observation_space,
            action_space=self.action_space,
            config=policy_config,
            model=model,
            loss=None,
            action_distribution_class=TorchDiagGaussian,
        )

        self.last_state = self.policy.get_initial_state()

    def get_config(self, config) -> Dict[str, any]:
        from ray.rllib.agents.ppo import DEFAULT_CONFIG

        policy_config = DEFAULT_CONFIG.copy()
        policy_config.update(config)
        return policy_config

    @property
    def action_space(self):
        return spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

    @property
    def observation_space(self):
        return spaces.Dict(
            {
                "depth": spaces.Box(
                    low=0, high=1, shape=(192, 256, 1), dtype=np.float64
                ),
                "image": spaces.Box(
                    low=0, high=1, shape=(192, 256, 3), dtype=np.float64
                ),
            }
        )

    def forward(self, input_dict: Dict[str, np.ndarray]):
        action, state = self._forward(input_dict, self.last_state)
        self.last_state = state
        return action

    def _forward(
        self, input_dict: Dict[str, np.ndarray], state: List[torch.Tensor]
    ) -> Tuple[np.ndarray, List[torch.Tensor]]:
        input_obs = self.preprocessor.transform(input_dict).reshape(1, -1)
        input_obs = torch.tensor(input_obs)

        action, state, action_info = self.policy.compute_actions(
            input_obs.cpu(), [s.reshape(1, -1) for s in state], explore=True
        )
        action = action.clip(self.action_space.low, self.action_space.high).reshape(2)
        return action, state

    def reset(self) -> None:
        self.last_state = self.policy.get_initial_state()
