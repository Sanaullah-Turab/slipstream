from __future__ import annotations

from typing import Any, Iterable, Optional, Sequence, Type, Union

import numpy as np
from gymnasium import Wrapper
from stable_baselines3.common.vec_env import VecEnv

from .multi_racing_env import MultiRacingEnv, AGENTS


class TwoCarVecEnv(VecEnv):
    """SB3 VecEnv wrapping one MultiRacingEnv as two parallel slots.

    slot 0 -> agent_0 | slot 1 -> agent_1
    """

    def __init__(self) -> None:
        self.env = MultiRacingEnv()
        obs_space = self.env.observation_space(AGENTS[0])
        act_space = self.env.action_space(AGENTS[0])
        super().__init__(num_envs=2, observation_space=obs_space, action_space=act_space)
        self._actions: Optional[np.ndarray] = None
        self._last_obs: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        obs_dict, _ = self.env.reset()
        self._last_obs = np.stack([obs_dict[a] for a in AGENTS])
        return self._last_obs

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
            obs_reset_dict, _ = self.env.reset()
            obs = np.stack([obs_reset_dict[a] for a in AGENTS])

        self._last_obs = obs
        self._actions = None
        return obs, rewards, dones, infos

    # ------------------------------------------------------------------
    # Required abstract methods
    # ------------------------------------------------------------------

    def close(self) -> None:
        self.env.close()

    def get_attr(self, attr_name: str, indices=None) -> list[Any]:
        val = getattr(self.env, attr_name)
        n = len(self._resolve_indices(indices))
        return [val] * n

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
        n = len(self._resolve_indices(indices))
        return [result] * n

    def env_is_wrapped(
        self, wrapper_class: Type[Wrapper], indices=None
    ) -> list[bool]:
        n = len(self._resolve_indices(indices))
        return [False] * n

    def seed(self, seed: Optional[int] = None) -> Sequence[Optional[int]]:
        self.env.reset(seed=seed)
        return [seed, seed]

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    def _resolve_indices(self, indices) -> list[int]:
        if indices is None:
            return list(range(self.num_envs))
        if isinstance(indices, int):
            return [indices]
        return list(indices)
