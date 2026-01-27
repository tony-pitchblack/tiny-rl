from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import minari
import mlflow
import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from tqdm import tqdm


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class GRUPolicy(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden_size: int = 128):
        super().__init__()
        self.rnn = nn.GRU(input_size=obs_dim, hidden_size=hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, n_actions)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        h, _ = self.rnn(obs)
        return self.head(h)


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("dataset_config must be a mapping")
    return cfg


def _flatten_cfg(cfg: Any, prefix: str) -> dict[str, str]:
    out: dict[str, str] = {}
    if isinstance(cfg, dict):
        for k, v in cfg.items():
            out.update(_flatten_cfg(v, f"{prefix}.{k}"))
        return out
    if isinstance(cfg, (list, tuple)):
        out[prefix] = yaml.safe_dump(cfg, default_flow_style=True).strip()
        return out
    out[prefix] = str(cfg)
    return out


def _require_mlflow_from_env() -> str:
    load_dotenv()
    host = os.environ["MLFLOW_HOST"]
    port = os.environ["MLFLOW_PORT"]
    uri = f"http://{host}:{port}"
    r = requests.get(f"{uri}/health", timeout=3)
    r.raise_for_status()
    return uri


def collate_fn(batch):
    obs_list, actions_list, lengths = [], [], []

    for ep in batch:
        obs = np.asarray(ep.observations, dtype=np.float32)
        T = len(ep.actions)
        obs = obs[:T]
        obs_list.append(torch.from_numpy(obs))
        actions_list.append(torch.as_tensor(ep.actions, dtype=torch.long))
        lengths.append(T)

    B = len(batch)
    max_T = max(lengths)
    obs_dim = int(obs_list[0].shape[-1])

    observations = torch.zeros((B, max_T, obs_dim), dtype=torch.float32)
    actions = torch.full((B, max_T), -1, dtype=torch.long)
    mask = torch.zeros((B, max_T), dtype=torch.bool)

    for i, (obs_i, act_i, T_i) in enumerate(zip(obs_list, actions_list, lengths)):
        observations[i, :T_i] = obs_i
        actions[i, :T_i] = act_i
        mask[i, :T_i] = True

    return {
        "observations": observations,
        "actions": actions,
        "mask": mask,
        "lengths": torch.as_tensor(lengths, dtype=torch.long),
    }


def make_train_val_masks(lengths: torch.Tensor, max_T: int, frac_val: float = 0.1):
    lengths = lengths.to(torch.long)
    B = int(lengths.shape[0])
    train_mask = torch.zeros((B, max_T), dtype=torch.bool)
    val_mask = torch.zeros((B, max_T), dtype=torch.bool)

    for i in range(B):
        T_i = int(lengths[i].item())
        split = int(np.floor((1.0 - frac_val) * T_i))
        split = max(0, min(split, T_i))
        train_mask[i, :split] = True
        val_mask[i, split:T_i] = True

    return train_mask, val_mask


def visualize_action_cells_on_val(
    *,
    model: nn.Module,
    dataset,
    n_actions: int,
    frac_val: float,
    seed: int,
    device: torch.device,
    epoch: int | None = None,
    state_dict_path: str | Path | None = None,
    state_dict: dict[str, torch.Tensor] | None = None,
    grid_height: int = 2,
    grid_width: int = 2,
) -> plt.Figure:
    was_training = model.training
    restore_state_dict = None

    if state_dict_path is not None or state_dict is not None:
        restore_state_dict = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if state_dict is None:
            state_dict = torch.load(state_dict_path, map_location=next(model.parameters()).device)
        model.load_state_dict(state_dict)

    model.eval()
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(dataset), size=grid_height * grid_width, replace=False)

    fig, axes = plt.subplots(grid_height, grid_width, figsize=(5 * grid_width, 4 * grid_height), squeeze=False)
    cmap = plt.matplotlib.colors.ListedColormap(["white", "C0", "C1", "C2"])
    norm = plt.matplotlib.colors.BoundaryNorm([0, 1, 2, 3, 4], cmap.N)

    if epoch is not None:
        fig.suptitle(f"epoch={epoch}")

    for ax, idx in zip(axes.ravel(), idxs):
        ep = dataset[int(idx)]

        obs_all = np.asarray(ep.observations, dtype=np.float32)
        T = int(len(ep.actions))
        if obs_all.shape[0] < T + 1:
            T = max(0, obs_all.shape[0] - 1)

        obs = obs_all[:T]
        target_actions = np.asarray(ep.actions, dtype=np.int64)[:T]

        with torch.no_grad():
            obs_t = torch.from_numpy(obs).view(1, T, -1).to(device)
            logits = model(obs_t)[0]
            pred_actions = torch.argmax(logits, dim=-1).cpu().numpy().astype(np.int64)

        split = int(np.floor((1.0 - frac_val) * T))
        split = max(0, min(split, T))
        val_T = T - split

        grid = np.zeros((n_actions, val_T), dtype=np.uint8)
        for j, t in enumerate(range(split, T)):
            a = int(target_actions[t])
            p = int(pred_actions[t])
            if a == p:
                grid[a, j] = 3
            else:
                grid[a, j] = 1
                grid[p, j] = 2

        ax.imshow(grid, aspect="auto", origin="lower", interpolation="nearest", cmap=cmap, norm=norm)
        ax.set_title(f"episode={int(idx)} val_len={val_T}")
        ax.set_xlabel("val time")
        ax.set_ylabel("action bin")

    import matplotlib.patches as mpatches

    legend_handles = [
        mpatches.Patch(color="C0", label="target"),
        mpatches.Patch(color="C1", label="pred"),
        mpatches.Patch(color="C2", label="match"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=3)
    fig.tight_layout(rect=(0, 0, 1, 0.92))

    if restore_state_dict is not None:
        model.load_state_dict(restore_state_dict)
    model.train(was_training)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_config", type=str, required=True)
    args = parser.parse_args()

    tracking_uri = _require_mlflow_from_env()
    mlflow.set_tracking_uri(tracking_uri)

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    seed = 42
    seed_everything(seed)

    dataset_cfg_path = Path(args.dataset_config)
    dataset_cfg = _load_yaml(dataset_cfg_path)

    root = Path(__file__).resolve().parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    import gymnasium as gym

    import datasets.discrete.envs.sine_waves  # noqa: F401

    dataset_id = str(dataset_cfg["dataset_id"])
    if dataset_id not in minari.list_local_datasets():
        raise FileNotFoundError(
            f"Minari dataset {dataset_id} not found locally. Build it with: "
            f"python datasets/discrete/build_dataset/sine_waves.py --config {dataset_cfg_path.as_posix()}"
        )

    ds = minari.load_dataset(dataset_id)
    loader_rng = torch.Generator()
    loader_rng.manual_seed(seed)
    dataloader = DataLoader(
        ds,
        batch_size=256,
        shuffle=True,
        collate_fn=collate_fn,
        generator=loader_rng,
    )

    env_cfg = dataset_cfg.get("env", {})
    if not isinstance(env_cfg, dict):
        raise ValueError("dataset_config.env must be a mapping")
    env_tmp = gym.make("SineWaves-v0", **env_cfg)
    obs_dim = int(np.prod(env_tmp.observation_space.shape))
    n_actions = int(env_tmp.action_space.n)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config_name = dataset_cfg_path.stem
    dataset_name = dataset_id.split("/", 1)[1] if "/" in dataset_id else dataset_id
    best_model_path = Path("data") / "experiments" / "discrete" / dataset_name / config_name / "best_model.pth"
    best_model_path.parent.mkdir(parents=True, exist_ok=True)

    mlflow.set_experiment(f"bc_seq2seq_discrete/{dataset_id}")

    with mlflow.start_run(run_name=f"{dataset_name}/{config_name}"):
        mlflow.log_params(_flatten_cfg(dataset_cfg, "dataset_config"))

        bc = GRUPolicy(obs_dim=obs_dim, n_actions=n_actions).to(device)
        opt = torch.optim.AdamW(bc.parameters(), lr=3e-4)
        torch.save(bc.state_dict(), best_model_path)

        patience_epochs = 10
        n_epochs = 1000
        eval_every_n_epochs = 50
        frac_val = 0.1

        best_val_loss = float("inf")
        best_epoch = -1
        bad_epochs = 0

        try:
            steps_per_epoch = len(dataloader)
        except TypeError:
            steps_per_epoch = None

        if steps_per_epoch is None:
            pbar = tqdm(range(n_epochs), desc="epochs", dynamic_ncols=True)
        else:
            pbar = tqdm(total=n_epochs * steps_per_epoch, desc="train", dynamic_ncols=True)

        for epoch in range(n_epochs):
            bc.train()
            train_loss_sum = 0.0
            train_n = 0
            val_loss_sum = 0.0
            val_n = 0

            for batch in dataloader:
                obs = batch["observations"].to(device)
                actions = batch["actions"].to(device)
                mask = batch["mask"].to(device)
                lengths = batch["lengths"]

                _, max_T, _ = obs.shape
                train_mask, val_mask = make_train_val_masks(lengths, max_T=max_T, frac_val=frac_val)
                train_mask = (train_mask.to(device) & mask)
                val_mask = (val_mask.to(device) & mask)

                logits = bc(obs)
                train_logits = logits[train_mask]
                train_actions = actions[train_mask]
                loss = F.cross_entropy(train_logits, train_actions)

                opt.zero_grad()
                loss.backward()
                opt.step()

                with torch.no_grad():
                    n_train = int(train_actions.numel())
                    train_loss_sum += float(loss.item()) * n_train
                    train_n += n_train

                    val_logits = logits[val_mask]
                    val_actions = actions[val_mask]
                    if val_actions.numel() > 0:
                        vloss = F.cross_entropy(val_logits, val_actions)
                        n_val = int(val_actions.numel())
                        val_loss_sum += float(vloss.item()) * n_val
                        val_n += n_val

                if steps_per_epoch is not None:
                    pbar.update(1)
                    pbar.set_postfix_str(
                        f"epoch={epoch} train={train_loss_sum / max(train_n, 1):.4f} "
                        f"val={val_loss_sum / max(val_n, 1):.4f}"
                    )

            train_loss = train_loss_sum / max(train_n, 1)
            val_loss = val_loss_sum / max(val_n, 1)

            mlflow.log_metric("train/loss", train_loss, step=epoch)
            mlflow.log_metric("val/loss", val_loss, step=epoch)

            if steps_per_epoch is None:
                pbar.set_postfix_str(f"train={train_loss:.4f} val={val_loss:.4f} bad={bad_epochs}")
                pbar.update(1)
            else:
                pbar.set_postfix_str(f"epoch={epoch} train={train_loss:.4f} val={val_loss:.4f} bad={bad_epochs}")

            improved = val_loss < best_val_loss
            if improved:
                best_val_loss = val_loss
                best_epoch = epoch
                bad_epochs = 0
                torch.save(bc.state_dict(), best_model_path)
            else:
                bad_epochs += 1

            if epoch % eval_every_n_epochs == 0:
                fig = visualize_action_cells_on_val(
                    model=bc,
                    dataset=ds,
                    n_actions=n_actions,
                    frac_val=frac_val,
                    seed=seed,
                    epoch=epoch,
                    device=device,
                )
                mlflow.log_figure(fig, f"viz/current/epoch_{epoch:05d}.png")
                plt.close(fig)

            if bad_epochs >= patience_epochs:
                break

        pbar.close()

        fig_best = visualize_action_cells_on_val(
            model=bc,
            dataset=ds,
            n_actions=n_actions,
            frac_val=frac_val,
            seed=seed,
            epoch=best_epoch,
            state_dict_path=best_model_path,
            device=device,
        )
        mlflow.log_figure(fig_best, "viz/best.png")
        plt.close(fig_best)


if __name__ == "__main__":
    main()

