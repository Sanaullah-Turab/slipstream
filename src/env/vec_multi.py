from __future__ import annotations

from typing import Any, Optional, Sequence, Type

import numpy as np
from gymnasium import Wrapper
from stable_baselines3.common.vec_env import VecEnv

from .multi_racing_env import MultiRacingEnv, AGENTS


class TwoCarVecEnv(VecEnv):
    """SB3 VecEnv wrapping one MultiRacingEnv as two parallel slots.

    slot 0 -> agent_0 | slot 1 -> agent_1
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self.env = MultiRacingEnv()
        self.base_seed = seed
        self.episode_count = 0
        obs_space = self.env.observation_space(AGENTS[0])
        act_space = self.env.action_space(AGENTS[0])
        super().__init__(num_envs=2, observation_space=obs_space, action_space=act_space)
        self._actions: Optional[np.ndarray] = None

    def reset(self) -> np.ndarray:
        reset_seed = None if self.base_seed is None else self.base_seed + self.episode_count
        self.episode_count += 1
        obs_dict, _ = self.env.reset(seed=reset_seed)
        return np.stack([obs_dict[a] for a in AGENTS])

    def step_async(self, actions: np.ndarray) -> None:
        self._actions = actions

    def step_wait(self):
        assert self._actions is not None
        action_dict = {a: self._actions[i] for i, a in enumerate(AGENTS)}
        obs_dict, rew_dict, term_dict, trunc_dict, info_dict = self.env.step(action_dict)

        obs = np.stack([obs_dict[a] for a in AGENTS])
        rewards = np.array([rew_dict[a] for a in AGENTS], dtype=np.float32)
        dones = np.array(
            [term_dict[a] or trunc_dict[a] for a in AGENTS], dtype=bool
        )

        infos = []
        for i, agent in enumerate(AGENTS):
            info = info_dict[agent].copy()
            if dones[i]:
                info["terminal_observation"] = obs[i].copy()
                info["TimeLimit.truncated"] = trunc_dict[agent]
            infos.append(info)

        if dones.any():
            reset_seed = None if self.base_seed is None else self.base_seed + self.episode_count
            self.episode_count += 1
            obs_reset_dict, _ = self.env.reset(seed=reset_seed)
            obs = np.stack([obs_reset_dict[a] for a in AGENTS])

        self._actions = None
        return obs, rewards, dones, infos

    def close(self) -> None:
        self.env.close()

    def get_attr(self, attr_name: str, indices=None) -> list[Any]:
        val = getattr(self.env, attr_name)
        return [val] * len(self._resolve_indices(indices))

    def set_attr(self, attr_name: str, value: Any, indices=None) -> None:
        setattr(self.env, attr_name, value)

    def env_method(
        self,
        method_name: str,
        *method_args,
        indices=None,
        **method_kwargs,
    ) -> list[Any]:
        result = getattr(self.env, method_name)(*method_args, **method_kwargs)
        return [result] * len(self._resolve_indices(indices))

    def env_is_wrapped(
        self, wrapper_class: Type[Wrapper], indices=None
    ) -> list[bool]:
        return [False] * len(self._resolve_indices(indices))

    def seed(self, seed: Optional[int] = None) -> Sequence[Optional[int]]:
        self.base_seed = seed
        self.episode_count = 0
        self.env.reset(seed=seed)
        return [seed, seed]

    def _resolve_indices(self, indices) -> list[int]:
        if indices is None:
            return list(range(self.num_envs))
        if isinstance(indices, int):
            return [indices]
        return list(indices)
