from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.envs.registration import register


@dataclass(frozen=True)
class SineWaveParams:
    length: int
    amplitude: float
    frequency: float
    phase: float
    dt: float


class SineWavesEnv(gym.Env[np.ndarray, int]):
    metadata = {}

    def __init__(
        self,
        *,
        min_length: int = 50,
        max_length: int = 200,
        dt: float = 1.0,
        min_amplitude: float = 0.5,
        max_amplitude: float = 2.0,
        n_amp_levels: int = 8,
        min_frequency: float = 0.2,
        max_frequency: float = 2.0,
        n_freq_levels: int = 8,
        n_action_bins: int = 101,
    ):
        if min_length < 2:
            raise ValueError("min_length must be >= 2")
        if max_length < min_length:
            raise ValueError("max_length must be >= min_length")
        if n_action_bins < 2:
            raise ValueError("n_action_bins must be >= 2")
        if n_amp_levels < 1:
            raise ValueError("n_amp_levels must be >= 1")
        if n_freq_levels < 1:
            raise ValueError("n_freq_levels must be >= 1")
        if max_amplitude <= 0:
            raise ValueError("max_amplitude must be > 0")
        if min_amplitude <= 0:
            raise ValueError("min_amplitude must be > 0")

        self.min_length = int(min_length)
        self.max_length = int(max_length)
        self.dt = float(dt)
        self.min_amplitude = float(min_amplitude)
        self.max_amplitude = float(max_amplitude)
        self.n_amp_levels = int(n_amp_levels)
        self.min_frequency = float(min_frequency)
        self.max_frequency = float(max_frequency)
        self.n_freq_levels = int(n_freq_levels)
        self.n_action_bins = int(n_action_bins)

        self._amp_levels = np.linspace(
            self.min_amplitude, self.max_amplitude, self.n_amp_levels, dtype=np.float32
        )
        self._freq_levels = np.linspace(
            self.min_frequency, self.max_frequency, self.n_freq_levels, dtype=np.float32
        )

        self.action_grid = np.linspace(
            -self.max_amplitude, self.max_amplitude, self.n_action_bins, dtype=np.float32
        )

        self.action_space = spaces.Discrete(self.n_action_bins)
        self.observation_space = spaces.Box(
            low=np.array([0.0, -self.max_amplitude], dtype=np.float32),
            high=np.array([1.0, self.max_amplitude], dtype=np.float32),
            dtype=np.float32,
        )

        self._params: SineWaveParams | None = None
        self._t_idx: int | None = None
        self._y: np.ndarray | None = None

    def _get_obs(self) -> np.ndarray:
        assert self._params is not None and self._t_idx is not None and self._y is not None
        t_norm = float(self._t_idx) / float(self._params.length - 1)
        return np.array([t_norm, float(self._y[self._t_idx])], dtype=np.float32)

    def _get_info(self) -> dict[str, Any]:
        assert self._params is not None and self._t_idx is not None
        return {
            "amplitude": self._params.amplitude,
            "frequency": self._params.frequency,
            "phase": self._params.phase,
            "length": self._params.length,
            "dt": self._params.dt,
            "t": int(self._t_idx),
        }

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)

        assert self.np_random is not None
        length = int(self.np_random.integers(self.min_length, self.max_length + 1))
        amplitude = float(self.np_random.choice(self._amp_levels))
        frequency = float(self.np_random.choice(self._freq_levels))
        phase = float(self.np_random.uniform(0.0, 2.0 * np.pi))

        self._params = SineWaveParams(
            length=length, amplitude=amplitude, frequency=frequency, phase=phase, dt=self.dt
        )
        t = (np.arange(length, dtype=np.float32) * float(self.dt)).astype(np.float32)
        self._y = (amplitude * np.sin(frequency * t + phase)).astype(np.float32)
        self._t_idx = 0

        return self._get_obs(), self._get_info()

    def oracle_action(self) -> int:
        assert self._params is not None and self._t_idx is not None and self._y is not None
        if self._t_idx >= self._params.length - 1:
            raise RuntimeError("oracle_action called at terminal state")
        target = float(self._y[self._t_idx + 1])
        return int(np.argmin((self.action_grid - target) ** 2))

    def step(self, action: int):
        assert self._params is not None and self._t_idx is not None and self._y is not None
        if self._t_idx >= self._params.length - 1:
            raise RuntimeError("step called after episode termination")

        y_hat = float(self.action_grid[int(action)])
        y_next = float(self._y[self._t_idx + 1])
        reward = -float((y_hat - y_next) ** 2)

        self._t_idx += 1
        terminated = self._t_idx >= self._params.length - 1

        return self._get_obs(), reward, terminated, False, self._get_info()


register(
    id="SineWaves-v0",
    entry_point="datasets.discrete.envs.sine_waves:SineWavesEnv",
)

