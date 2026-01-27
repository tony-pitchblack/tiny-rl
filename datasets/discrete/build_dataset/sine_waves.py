from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import minari
import yaml
from minari.storage.datasets_root_dir import get_dataset_path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import datasets.discrete.envs.sine_waves  # noqa: F401


def _load_cfg(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a mapping")
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="datasets/discrete/cfg/sine_waves/default.yml",
    )
    parser.add_argument("--dataset-id", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = _load_cfg(args.config)
    env_cfg = cfg.get("env", {})
    if not isinstance(env_cfg, dict):
        raise ValueError("env must be a mapping")

    dataset_id = str(args.dataset_id or cfg.get("dataset_id"))
    dataset_root = get_dataset_path(dataset_id)
    if dataset_root.exists():
        if not args.overwrite:
            raise SystemExit(
                f"Dataset {dataset_id} already exists at {dataset_root}. Use --overwrite to replace."
            )
        shutil.rmtree(dataset_root)
    episodes = int(cfg.get("episodes", 100))
    seed = int(cfg.get("seed", 0))
    data_format = cfg.get("data_format", None)

    env = gym.make("SineWaves-v0", **env_cfg)
    collector = minari.DataCollector(
        env,
        data_format=data_format,
        record_infos=True,
    )

    for ep in range(episodes):
        collector.reset(seed=seed + ep)
        while True:
            action = collector.unwrapped.oracle_action()
            _, _, terminated, truncated, _ = collector.step(action)
            if terminated or truncated:
                break

    collector.create_dataset(
        dataset_id=dataset_id,
        algorithm_name="SineOracle",
        description="Discrete sine wave one-step prediction with oracle actions.",
    )


if __name__ == "__main__":
    main()

