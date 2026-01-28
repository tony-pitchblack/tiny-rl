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
import pandas as pd
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


class TokenGRUPolicy(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int = 128, emb_dim: int = 64):
        super().__init__()
        self.emb = nn.Embedding(vocab_size + 1, emb_dim)
        self.rnn = nn.GRU(input_size=emb_dim, hidden_size=hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.emb(input_ids)
        h, _ = self.rnn(x)
        return self.head(h)


class GRU4Rec(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int = 128, emb_dim: int = 64):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.emb = nn.Embedding(self.vocab_size, emb_dim)
        self.rnn = nn.GRU(input_size=emb_dim, hidden_size=hidden_size, batch_first=True)
        self.out = nn.Embedding(self.vocab_size, hidden_size)

    def step(self, input_ids: torch.Tensor, h: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.emb(input_ids)
        y, h = self.rnn(x, h)
        return y[:, 0, :], h

    def score_items(self, h_t: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        v = self.out(item_ids)
        return (h_t * v).sum(dim=-1)


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


def collate_token_sequences(batch: list[torch.Tensor], pad_token: int) -> dict[str, torch.Tensor]:
    lengths = [int(x.numel()) for x in batch]
    B = len(batch)
    max_L = max(lengths)
    tokens = torch.full((B, max_L), pad_token, dtype=torch.long)
    for i, (seq, L) in enumerate(zip(batch, lengths)):
        tokens[i, :L] = seq
    return {"tokens": tokens}


def _build_sa2c_sequences(
    df: pd.DataFrame,
    *,
    max_seq_len: int,
    item2id: dict[int, int],
) -> list[torch.Tensor]:
    required = {"timestamp", "session_id", "item_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"SA2C_code dataframe missing columns: {sorted(missing)}")

    df = df.loc[:, ["timestamp", "session_id", "item_id"]].copy()
    df["item_id"] = df["item_id"].astype(np.int64)
    df["session_id"] = df["session_id"].astype(np.int64)
    df = df.sort_values(["session_id", "timestamp"], kind="mergesort")

    sequences: list[torch.Tensor] = []
    window = int(max_seq_len) + 1
    for _, g in df.groupby("session_id", sort=False):
        items = g["item_id"].to_numpy(dtype=np.int64, copy=False)
        if items.size < 2:
            continue
        if items.size > window:
            items = items[-window:]
        mapped = np.fromiter((item2id[int(x)] for x in items), dtype=np.int64, count=items.size)
        sequences.append(torch.from_numpy(mapped))
    return sequences


def _load_sa2c_pickles(dataset_cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_dir_cfg = dataset_cfg.get("data_dir")
    train_file = str(dataset_cfg.get("train_file", "sampled_train.df"))
    val_file = str(dataset_cfg.get("val_file", "sampled_val.df"))

    repo_root = Path(__file__).resolve().parent
    if data_dir_cfg is None:
        data_dir = repo_root / "SA2C_code" / "Kaggle" / "data"
    else:
        data_dir = Path(str(data_dir_cfg)).expanduser()
        if not data_dir.is_absolute():
            data_dir = (repo_root / data_dir).resolve()

    train_path = data_dir / train_file
    val_path = data_dir / val_file
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(f"Missing SA2C_code pickles: {train_path.as_posix()} or {val_path.as_posix()}")

    return pd.read_pickle(train_path), pd.read_pickle(val_path)


def _iter_session_parallel_steps(
    sequences: list[torch.Tensor],
    *,
    batch_size: int,
    seed: int,
    shuffle: bool,
    device: torch.device,
):
    if len(sequences) == 0:
        return

    order = np.arange(len(sequences), dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    if shuffle:
        rng.shuffle(order)

    next_ptr = 0
    B = min(int(batch_size), int(len(sequences)))
    batch_idxs = order[:B].tolist()
    next_ptr = B
    positions = [0] * B
    reset = torch.zeros((B,), dtype=torch.bool, device=device)

    while B > 0:
        x = torch.empty((B, 1), dtype=torch.long, device=device)
        y = torch.empty((B,), dtype=torch.long, device=device)
        reset.fill_(False)

        for i in range(B):
            s = sequences[int(batch_idxs[i])]
            p = int(positions[i])
            x[i, 0] = int(s[p].item())
            y[i] = int(s[p + 1].item())
            positions[i] = p + 1

        ended: list[int] = []
        for i in range(B):
            s = sequences[int(batch_idxs[i])]
            if int(positions[i]) + 1 >= int(s.numel()):
                ended.append(i)

        drop: set[int] = set()
        for i in ended:
            if next_ptr < len(order):
                batch_idxs[i] = int(order[next_ptr])
                next_ptr += 1
                positions[i] = 0
                reset[i] = True
            else:
                drop.add(i)

        yield x, y, reset, drop

        if drop:
            keep = [i for i in range(B) if i not in drop]
            batch_idxs = [batch_idxs[i] for i in keep]
            positions = [positions[i] for i in keep]
            B = len(keep)
            reset = reset[keep]


def _inbatch_negative_targets(targets: torch.Tensor, *, seed: int) -> torch.Tensor:
    B = int(targets.shape[0])
    if B <= 1:
        return targets.clone()
    g = torch.Generator(device=targets.device)
    g.manual_seed(int(seed))
    perm = torch.randperm(B, generator=g, device=targets.device)
    neg = targets[perm]
    same = neg.eq(targets)
    if same.any():
        perm2 = (perm + 1) % B
        neg = torch.where(same, targets[perm2], neg)
    return neg


def _ndcg_at_k_from_pos_neg(pos_scores: torch.Tensor, neg_scores: torch.Tensor, k: int) -> float:
    if pos_scores.numel() == 0:
        return 0.0
    if neg_scores.numel() == 0:
        return 1.0
    rank = 1 + neg_scores.ge(pos_scores.unsqueeze(1)).sum(dim=1)
    rank = rank.to(torch.float32)
    k = float(max(1, int(k)))
    dcg = torch.where(rank <= k, 1.0 / torch.log2(rank + 1.0), torch.zeros_like(rank))
    return float(dcg.mean().item())


def _ndcg_at_k_one_relevant(logits: torch.Tensor, targets: torch.Tensor, k: int = 10) -> float:
    if targets.numel() == 0:
        return 0.0
    k = min(int(k), int(logits.shape[-1]))
    topk = torch.topk(logits, k=k, dim=-1).indices
    matches = topk.eq(targets.unsqueeze(1))
    any_match = matches.any(dim=1)
    ranks0 = matches.float().argmax(dim=1) + 1
    ranks = torch.where(any_match, ranks0, torch.zeros_like(ranks0))
    dcg = torch.where(
        ranks > 0,
        1.0 / torch.log2(ranks.to(torch.float32) + 1.0),
        torch.zeros_like(ranks, dtype=torch.float32),
    )
    return float(dcg.mean().item())


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

    dataset_id = str(dataset_cfg["dataset_id"])
    is_sa2c_code_dataset = bool(dataset_cfg.get("is_sa2c_code_dataset", False))
    batch_size = int(dataset_cfg.get("batch_size", 256))

    ds = None
    dataloader = None
    val_dataloader = None
    train_seqs = None
    val_seqs = None
    obs_dim = None
    n_actions = None
    vocab_size = None
    pad_token = None

    if is_sa2c_code_dataset:
        max_seq_len = int(dataset_cfg.get("max_seq_len", 50))
        train_df, val_df = _load_sa2c_pickles(dataset_cfg)

        all_items = pd.concat([train_df["item_id"], val_df["item_id"]], ignore_index=True).astype(np.int64)
        uniq = np.sort(all_items.unique())
        item2id = {int(x): int(i) for i, x in enumerate(uniq)}
        vocab_size = len(item2id)
        train_seqs = _build_sa2c_sequences(train_df, max_seq_len=max_seq_len, item2id=item2id)
        val_seqs = _build_sa2c_sequences(val_df, max_seq_len=max_seq_len, item2id=item2id)
        if len(train_seqs) == 0 or len(val_seqs) == 0:
            raise ValueError("No valid (len>=2) sessions found in SA2C_code pickles")
    else:
        import gymnasium as gym

        import datasets.discrete.envs.sine_waves  # noqa: F401

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
            batch_size=batch_size,
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

        patience_epochs = int(dataset_cfg.get("patience_epochs", 10))
        n_epochs = int(dataset_cfg.get("n_epochs", 1000))
        eval_every_n_epochs = int(dataset_cfg.get("eval_every_n_epochs", 50))
        frac_val = 0.1

        best_val_loss = float("inf")
        best_epoch = -1
        bad_epochs = 0

        if is_sa2c_code_dataset:
            assert vocab_size is not None
            assert train_seqs is not None and val_seqs is not None

            emb_dim = int(dataset_cfg.get("emb_dim", 64))
            hidden_size = int(dataset_cfg.get("hidden_size", 128))
            lr = float(dataset_cfg.get("lr", 3e-4))
            ndcg_k = int(dataset_cfg.get("ndcg_k", 10))

            model = GRU4Rec(vocab_size=vocab_size, hidden_size=hidden_size, emb_dim=emb_dim).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=lr)
            torch.save(model.state_dict(), best_model_path)

            pbar = tqdm(range(n_epochs), desc="epochs", dynamic_ncols=True)
            for epoch in range(n_epochs):
                model.train()
                train_loss_sum = 0.0
                train_n = 0
                step_seed = seed + 1000 * epoch

                h = None
                pending_reset = None
                pending_drop: set[int] | None = None
                step_idx = 0
                for x, y, reset, drop in _iter_session_parallel_steps(
                    train_seqs, batch_size=batch_size, seed=step_seed, shuffle=True, device=device
                ):
                    if h is not None:
                        if pending_drop:
                            keep = [i for i in range(h.shape[1]) if i not in pending_drop]
                            h = h[:, keep, :]
                        if pending_reset is not None and bool(pending_reset.any().item()):
                            h = h.clone()
                            h[:, pending_reset, :] = 0

                    h_t, h = model.step(x, h)
                    neg = _inbatch_negative_targets(y, seed=step_seed + step_idx)
                    pos_scores = model.score_items(h_t, y)
                    neg_scores = model.score_items(h_t, neg)
                    loss = -F.logsigmoid(pos_scores - neg_scores).mean()

                    opt.zero_grad()
                    loss.backward()
                    opt.step()

                    with torch.no_grad():
                        B = int(y.shape[0])
                        train_loss_sum += float(loss.item()) * B
                        train_n += B

                    if drop:
                        keep = [i for i in range(int(reset.shape[0])) if i not in drop]
                        pending_reset = reset[keep]
                    else:
                        pending_reset = reset
                    pending_drop = drop
                    step_idx += 1

                model.eval()
                val_loss_sum = 0.0
                val_n = 0
                val_ndcg_sum = 0.0
                val_ndcg_n = 0

                with torch.no_grad():
                    h = None
                    pending_reset = None
                    pending_drop = None
                    step_idx = 0
                    for x, y, reset, drop in _iter_session_parallel_steps(
                        val_seqs, batch_size=batch_size, seed=seed, shuffle=False, device=device
                    ):
                        if h is not None:
                            if pending_drop:
                                keep = [i for i in range(h.shape[1]) if i not in pending_drop]
                                h = h[:, keep, :]
                            if pending_reset is not None and bool(pending_reset.any().item()):
                                h = h.clone()
                                h[:, pending_reset, :] = 0

                        h_t, h = model.step(x, h)
                        neg = _inbatch_negative_targets(y, seed=seed + step_idx)
                        pos_scores = model.score_items(h_t, y)
                        neg_scores = model.score_items(h_t, neg)
                        vloss = -F.logsigmoid(pos_scores - neg_scores).mean()

                        B = int(y.shape[0])
                        val_loss_sum += float(vloss.item()) * B
                        val_n += B
                        val_ndcg_sum += _ndcg_at_k_from_pos_neg(pos_scores, neg_scores.unsqueeze(1), k=ndcg_k) * B
                        val_ndcg_n += B

                        if drop:
                            keep = [i for i in range(int(reset.shape[0])) if i not in drop]
                            pending_reset = reset[keep]
                        else:
                            pending_reset = reset
                        pending_drop = drop
                        step_idx += 1

                train_loss = train_loss_sum / max(train_n, 1)
                val_loss = val_loss_sum / max(val_n, 1)

                mlflow.log_metric("train/loss", train_loss, step=epoch)
                mlflow.log_metric("val/loss", val_loss, step=epoch)
                mlflow.log_metric(f"val/NDCG@{ndcg_k}", val_ndcg_sum / max(val_ndcg_n, 1), step=epoch)

                pbar.set_postfix_str(f"train={train_loss:.4f} val={val_loss:.4f} bad={bad_epochs}")
                pbar.update(1)

                improved = val_loss < best_val_loss
                if improved:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    bad_epochs = 0
                    torch.save(model.state_dict(), best_model_path)
                else:
                    bad_epochs += 1

                if bad_epochs >= patience_epochs:
                    break

            pbar.close()
        else:
            assert dataloader is not None
            assert obs_dim is not None
            bc = GRUPolicy(obs_dim=obs_dim, n_actions=n_actions).to(device)
            opt = torch.optim.AdamW(bc.parameters(), lr=3e-4)
            torch.save(bc.state_dict(), best_model_path)

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
                    train_targets = actions[train_mask]
                    loss = F.cross_entropy(train_logits, train_targets)

                    opt.zero_grad()
                    loss.backward()
                    opt.step()

                    with torch.no_grad():
                        n_train = int(train_targets.numel())
                        train_loss_sum += float(loss.item()) * n_train
                        train_n += n_train

                        val_logits = logits[val_mask]
                        val_targets = actions[val_mask]
                        if val_targets.numel() > 0:
                            vloss = F.cross_entropy(val_logits, val_targets)
                            n_val = int(val_targets.numel())
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

