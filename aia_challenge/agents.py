from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from gym import spaces


def get_all_subclasses(cls: object):
    """Get all subclasses of `cls`"""
    return set(cls.__subclasses__()).union(
        [s for c in cls.__subclasses__() for s in get_all_subclasses(c)]
    )


def get_subclass(class_name: str, base_cls: object) -> object:
    """Get `class_name` object, which is a subclass
    of `base_cls`"""
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
    def __init__(self, config: Dict[str, Any]) -> None:
        """Submission inferface.

        Submission must implement the `forward` method,
        which takes an observation and provides and action.
        The `reset` method may be used to set any episode
        specific configuration (e.g., RNN states).

        Parameters
        ----------
        config: Dict[str, Any]
            Configuration provided by the user.
            Configuration file is a yaml that is loaded
            by the evalaution script. See the challenge instructions
            for an example.
        """
        pass

    def act(self, input: Dict[str, np.ndarray]) -> np.ndarray:
        """Take an observation and provide an action.

        Parameters
        ----------
        input: Dict[str, np.ndarray]
            Dictionary of input observations. Keys are fields
            names and values are corresponding data.

        Returns
        -------
        np.ndarray, shape=(2,)
            Array of forward velociy and yaw rate commands.
        """
        raise NotImplementedError()

    def reset(self) -> None:
        """Called at the start of every episode, which
        allows for any episode specific configuration (e.g.,
        RNN state)."""
        pass


class RllibAgent(SearchAgent):
    def __init__(self, config: Dict[str, Any]) -> None:
        """Example submission implementation for the policy
        trained with the `train-agent.py` script.

        This policy is designed for inference only,
        and thus it doesn't require a flightgoggles
        instance, PPO configrations, or any of the other
        more resource-intensive componenets required for
        training.

        Parameters
        ----------
        config_path: Dict[str, Any]
            Configuration dictionary provided by the user.
        """
        from ray.rllib.models.preprocessors import DictFlatteningPreprocessor
        from ray.rllib.models.torch.torch_action_dist import TorchDiagGaussian
        from ray.rllib.policy import TorchPolicy
        from rllib_policies.vision import NatureCNNRNNActorCritic

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
        """Update default PPO config with custom options
        given by user.

        Parameters
        ----------
        config: Dict[str, Any]
            Dictionary of custom configurations.

        Returns
        -------
        Dict[str, Any]
            RLlib PPO configuration.
        """
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
                    low=0, high=1, shape=(192, 256, 1), dtype=np.float32
                ),
                "grayscale": spaces.Box(
                    low=0, high=1, shape=(192, 256, 1), dtype=np.float32
                ),
            }
        )

    def act(self, input_dict: Dict[str, np.ndarray]):
        """Provide forward velociy and yaw rate given
        an observation. This method wraps `_act`, which
        handles RNN state.

        Parameters
        ----------
        input: Dict[str, np.ndarray]
            Dictionary of input observations. Keys are fields
            names and values are corresponding data.

        Returns
        -------
        np.ndarray, shape=(2,)
            Array of forward velociy and yaw rate commands.
        """
        action, state = self._act(input_dict, self.last_state)
        self.last_state = state
        return action

    def _act(
        self, input_dict: Dict[str, np.ndarray], state: List[torch.Tensor]
    ) -> Tuple[np.ndarray, List[torch.Tensor]]:
        """Peform all steps required for policy inference including
        observation preprocessing, policy inference, and action
        postprocessing.

        Parameters
        ----------
        input: Dict[str, np.ndarray]
            Dictionary of input observations. Keys are fields
            names and values are corresponding data.
        state: List[torch.Tensor]
            RNN states.

        Returns
        -------
        Tuple[np.ndarray, Tuple[torch.Tensor, torch.Tensor]]
            - shape (2, ) array of forward velociy and yaw rate commands.
            - list of RNN states. LSTM states have two tensors, GRU states
              have one tensor.
        """
        input_obs = self.preprocessor.transform(input_dict).reshape(1, -1)
        input_obs = torch.tensor(input_obs)

        action, state, action_info = self.policy.compute_actions(
            input_obs.cpu(), [s.reshape(1, -1) for s in state], explore=True
        )
        action = action.clip(self.action_space.low, self.action_space.high).reshape(2)
        return action, state

    def reset(self) -> None:
        """Initialize RNN state."""
        self.last_state = self.policy.get_initial_state()


class RandomAgent(SearchAgent):
    """Agent that samples forward velocity and yaw rate
    from a 2D multivariate gaussian distribution."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.action_mean = np.array(config["action_mean"])
        self.action_cov = np.zeros((2, 2))
        np.fill_diagonal(self.action_cov, config["action_var"])

    def act(self, input: Dict[str, np.ndarray]) -> np.ndarray:
        action = np.random.multivariate_normal(self.action_mean, self.action_cov)
        return action.clip(-1, 1).astype(np.float32)
