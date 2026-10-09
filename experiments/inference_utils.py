"""Shared checkpoint loading and full-catalogue inference helpers."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from warprec.utils.registry import model_registry


def padding_side_for_model(model):
    return "right"


@dataclass(frozen=True)
class RankingResult:
    top_items: list[int]
    ndcg: float
    recall: float


def padded_sequences(
    sequences: list[list[int]],
    max_seq_len: int,
    padding_idx: int,
    padding_side: str = "right",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create fixed-width sequence tensors for one model inference batch."""
    tensor = torch.full((len(sequences), max_seq_len), padding_idx, dtype=torch.long)
    lengths = []
    for row_idx, sequence in enumerate(sequences):
        visible = sequence[-max_seq_len:]
        seq_len = len(visible)
        if seq_len == 0:
            visible = [padding_idx]
            seq_len = 1

        clipped = torch.tensor(visible[-max_seq_len:], dtype=torch.long)
        if padding_side == "left":
            tensor[row_idx, max_seq_len - clipped.numel() :] = clipped
        else:
            tensor[row_idx, : clipped.numel()] = clipped
        lengths.append(min(seq_len, max_seq_len))

    return tensor, torch.tensor(lengths, dtype=torch.long)


def ranking_metrics(
    top_items: Iterable[int], positives: set[int], k: int
) -> tuple[float, float]:
    """Compute binary nDCG@k and Recall@k for one user."""
    top_items = list(top_items)[:k]
    if not positives:
        return 0.0, 0.0

    dcg = 0.0
    for rank, item in enumerate(top_items, start=1):
        if item in positives:
            dcg += 1.0 / math.log2(rank + 1)

    ideal_hits = min(len(positives), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    recall = sum(1 for item in top_items if item in positives) / float(len(positives))
    return (dcg / idcg if idcg > 0 else 0.0), recall


def resolve_checkpoint(path_value: str | None) -> Path | None:
    """Resolve a checkpoint file from a config meta.load_from value."""
    if not path_value:
        return None

    path = Path(path_value).expanduser()
    if path.is_file():
        return path

    if path.is_dir():
        direct = path / "checkpoint.pt"
        if direct.is_file():
            return direct
        matches = sorted(path.rglob("checkpoint.pt"))
        if matches:
            return matches[-1]

    return path


def load_model_params(
    config_params: dict, checkpoint_path: Path | None, checkpoint: dict | None = None
) -> dict:
    """Prefer Ray Tune params.json when available, otherwise use concrete config values."""
    if checkpoint is not None and checkpoint.get("params"):
        return checkpoint["params"]

    if checkpoint_path is not None:
        trial_dir = checkpoint_path.parent.parent
        params_path = trial_dir / "params.json"
        if params_path.is_file():
            with params_path.open("r", encoding="utf-8") as handle:
                params = json.load(handle)
            return {k: v for k, v in params.items() if k != "meta"}

    params = {}
    for key, value in config_params.items():
        if key == "meta":
            continue
        if isinstance(value, list) and value:
            params[key] = value[-1]
        else:
            params[key] = value
    return params


def load_model(
    model_name: str, model_config: dict, dataset, device: torch.device, strict: bool
):
    """Instantiate a model and load the checkpoint configured for it."""
    checkpoint_path = resolve_checkpoint(model_config.get("meta", {}).get("load_from"))
    if checkpoint_path is None or not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"{model_name}: checkpoint not found. Expected meta.load_from to point to a checkpoint.pt file."
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_dict = checkpoint if isinstance(checkpoint, dict) else None
    if (
        checkpoint_dict is None
        or checkpoint_dict.get("checkpoint_selection") != "validation_ndcg10"
        or not checkpoint_dict.get("full_epoch")
    ):
        raise ValueError(
            "Expected a completed validation-selected checkpoint; retrain with the current pipeline"
        )
    params = load_model_params(model_config, checkpoint_path, checkpoint_dict)
    info = (
        checkpoint_dict.get("info", dataset.info())
        if checkpoint_dict is not None
        else dataset.info()
    )
    if checkpoint_dict is not None and "mapping_digest" in checkpoint_dict:
        import hashlib

        current_digest = hashlib.sha256(
            json.dumps(
                dataset.get_inverse_mappings(), sort_keys=True, default=str
            ).encode()
        ).hexdigest()
        if current_digest != checkpoint_dict["mapping_digest"]:
            raise ValueError("Training/evaluation user-item mapping mismatch")
    model = model_registry.get_class(model_name)(
        params=params,
        interactions=dataset.train_set,
        info=info,
    )

    if isinstance(checkpoint, dict):
        for key in ("model_state", "model_state_dict", "state_dict", "model"):
            if key in checkpoint:
                state_dict = checkpoint[key]
                break
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict, strict=strict)
    model.to(device)
    model.eval()
    return model


def get_train_histories(train_session) -> dict[int, list[int]]:
    """Return chronological item histories for each internal user id."""
    if hasattr(train_session, "_compute_cache_user_history"):
        train_session._compute_cache_user_history()
        return train_session._cached_user_histories

    cached = getattr(train_session, "_cached_user_histories", None)
    if cached:
        return cached

    flat_items = getattr(train_session, "_flat_items", None)
    user_offsets = getattr(train_session, "_user_offsets", None)
    if flat_items is not None and user_offsets is not None:
        histories = {}
        for user_idx, (start, end) in enumerate(
            zip(user_offsets[:-1], user_offsets[1:])
        ):
            if end > start:
                histories[user_idx] = flat_items[start:end].tolist()
        train_session._cached_user_histories = histories
        return histories

    if hasattr(train_session, "_get_user_sessions"):
        sessions = train_session._get_user_sessions()
        histories = {
            int(user_idx): [int(item) for item in items]
            for user_idx, items in sessions.items()
        }
        train_session._cached_user_histories = histories
        return histories

    raise AttributeError("Unable to build user histories from the train session.")


@torch.no_grad()
def recommend_batch(
    model,
    user_idxs: list[int],
    sequences: list[list[int]],
    mask_sequences: list[list[int]],
    positives_by_user: dict[int, set[int]],
    k: int,
    device: torch.device,
    compute_metrics: bool = True,
) -> list[RankingResult]:
    """Compute Top-k recommendation lists and effectiveness metrics for one batch."""
    user_indices = torch.tensor(user_idxs, dtype=torch.long, device=device)
    user_seq, seq_len = padded_sequences(
        sequences,
        model.max_seq_len,
        model.n_items,
        padding_side=padding_side_for_model(model),
    )
    scores = model.predict(
        user_indices=user_indices,
        user_seq=user_seq.to(device),
        seq_len=seq_len.to(device),
    )

    for row_idx, mask_sequence in enumerate(mask_sequences):
        if not mask_sequence:
            continue
        seen_items = torch.tensor(
            list(set(mask_sequence)), dtype=torch.long, device=device
        )
        scores[row_idx, seen_items] = -torch.inf

    values, indices = torch.topk(scores, min(k, scores.shape[1]), dim=1)
    top_items_batch = [
        ids[torch.isfinite(row)].detach().cpu().tolist()
        for row, ids in zip(values, indices)
    ]
    results = []
    for user_idx, top_items in zip(user_idxs, top_items_batch):
        ndcg, recall = (
            ranking_metrics(top_items, positives_by_user[user_idx], k)
            if compute_metrics
            else (float("nan"), float("nan"))
        )
        results.append(RankingResult(top_items=top_items, ndcg=ndcg, recall=recall))
    return results


def batched(values: list[int], batch_size: int) -> Iterable[list[int]]:
    """Yield consecutive chunks from a list."""
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def choose_users(
    histories: dict[int, list[int]],
    eval_sparse,
    min_history: int,
    max_users: int | None,
    seed: int,
    random_sample: bool,
) -> list[int]:
    """Select users that have enough history and at least one evaluation item."""
    users = [
        user_idx
        for user_idx, history in histories.items()
        if len(history) >= min_history
        and eval_sparse.indptr[user_idx + 1] > eval_sparse.indptr[user_idx]
    ]
    users = sorted(users)
    if max_users is None or max_users >= len(users):
        return users
    if random_sample:
        rng = np.random.default_rng(seed)
        return sorted(rng.choice(users, size=max_users, replace=False).tolist())
    return users[:max_users]
