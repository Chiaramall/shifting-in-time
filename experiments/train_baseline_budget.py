"""Train sequential models and select checkpoints by validation nDCG@10."""

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from inference_utils import batched, choose_users, get_train_histories, recommend_batch

from warprec.common import initialize_datasets, standard_optimizer
from warprec.data.reader import ReaderFactory
from warprec.utils.callback import WarpRecCallback
from warprec.utils.config import load_train_configuration
from warprec.utils.registry import model_registry

TRAINING_PROTOCOL = "validation_selected_v2"


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)


def atomic_checkpoint(path, state):
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def mapping_digest(dataset):
    return hashlib.sha256(
        json.dumps(dataset.get_inverse_mappings(), sort_keys=True, default=str).encode()
    ).hexdigest()


def validation_digest(dataset):
    matrix = dataset.eval_set.get_sparse().tocsr()
    digest = hashlib.sha256()
    for values in (matrix.indptr, matrix.indices, matrix.data):
        digest.update(values.tobytes())
    return digest.hexdigest()


def make_loader(model, dataset, params, epoch, generator):
    if model.__class__.__name__ == "gSASRec":
        from semantic_training import CompactHistoryDataset, HistoryCollator

        samples = CompactHistoryDataset(
            get_train_histories(dataset.train_session), model.max_seq_len, model.n_items
        )
        return torch.utils.data.DataLoader(
            samples,
            batch_size=params["batch_size"],
            shuffle=True,
            generator=generator,
            collate_fn=HistoryCollator(model.n_items, model.neg_samples),
            num_workers=0,
        )
    if model.__class__.__name__ == "BERT4Rec":
        key = f"cloze_len_{model.max_seq_len}_prob_{model.mask_prob}_neg_{model.neg_samples}"
        dataset.train_session._cached_dataset.pop(key, None)
        loader = dataset.train_session.get_cloze_mask_dataloader(
            max_seq_len=model.max_seq_len,
            mask_prob=model.mask_prob,
            mask_token_id=model.mask_token_id,
            neg_samples=model.neg_samples,
            batch_size=params["batch_size"],
            seed=42 + epoch,
            low_memory=False,
        )
    else:
        loader = model.get_dataloader(
            interactions=dataset.train_set,
            sessions=dataset.train_session,
            low_memory=False,
        )
    if model.__class__.__name__ == "DuoRec":
        from semantic_training import IndexDataset, SemanticCollator

        return torch.utils.data.DataLoader(
            IndexDataset(loader.dataset),
            batch_size=params["batch_size"],
            shuffle=True,
            generator=generator,
            collate_fn=SemanticCollator(loader.dataset),
            num_workers=0,
        )
    return torch.utils.data.DataLoader(
        loader.dataset,
        batch_size=params["batch_size"],
        shuffle=True,
        generator=generator,
        collate_fn=loader.collate_fn,
        num_workers=0,
    )


@torch.inference_mode()
def evaluate_validation(model, dataset, device, batch_size=64):
    """Full-catalogue nDCG@10 on validation positives, using training histories only."""
    model.eval()
    histories = get_train_histories(dataset.train_session)
    matrix = dataset.eval_set.get_sparse().tocsr()
    users = choose_users(histories, matrix, 1, None, 42, False)
    if not users:
        raise ValueError("No eligible validation users")
    positives = {
        u: set(matrix.indices[matrix.indptr[u] : matrix.indptr[u + 1]]) for u in users
    }
    total = 0.0
    for batch in batched(users, batch_size):
        results = recommend_batch(
            model,
            batch,
            [histories[u] for u in batch],
            [histories[u] for u in batch],
            positives,
            10,
            torch.device(device),
        )
        total += sum(result.ndcg for result in results)
    return total / len(users)


def train(
    config, dataset, name, out, smoke=False, stop_after=300, validation_dataset=None
):
    if validation_dataset is None:
        raise ValueError("Training requires a separate validation dataset")
    if mapping_digest(dataset) != mapping_digest(validation_dataset):
        raise ValueError("Training and validation item/user mappings differ")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    params = {
        key: value[-1] if isinstance(value, list) else value
        for key, value in dict(config.models[name]).items()
        if key not in ("meta", "optimization", "early_stopping", "lr_scheduler")
    }
    params["corrected_training"] = True
    target_epochs = int(params["epochs"])
    if target_epochs < 1 or stop_after < 1:
        raise ValueError("Epoch limits must be positive")
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    device = config.general.device
    model = model_registry.get_class(name)(
        params=params, interactions=dataset.train_set, info=dataset.info(), seed=42
    ).to(device)
    optimizer = standard_optimizer(model)
    generator = torch.Generator().manual_seed(42)
    loader = make_loader(model, dataset, params, 0, generator)
    if not len(loader):
        raise ValueError("No training batches")
    digest = mapping_digest(dataset)
    val_digest = validation_digest(validation_dataset)
    checkpoint = out / "checkpoint.pt"
    last_checkpoint = out / "last.pt"
    history = []
    start_epoch, best_epoch, best_score = 0, 0, -float("inf")
    if checkpoint.exists() and not last_checkpoint.exists():
        raise ValueError(
            "Checkpoint without resume state; use a fresh output directory"
        )
    if last_checkpoint.exists():
        state = torch.load(last_checkpoint, map_location="cpu", weights_only=False)
        expected = {
            "training_protocol": TRAINING_PROTOCOL,
            "params": params,
            "mapping_digest": digest,
            "validation_digest": val_digest,
            "model_name": name,
            "full_epoch": not smoke,
        }
        if any(state.get(key) != value for key, value in expected.items()):
            raise ValueError(
                "Resume state differs from training configuration or split"
            )
        model.load_state_dict(state["model_state"], strict=True)
        optimizer.load_state_dict(state["optimizer_state"])
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        generator.set_state(state["loader_rng"])
        start_epoch, history = state["epoch"], state["history"]
        best_epoch, best_score = state["best_epoch"], state["best_validation_ndcg10"]
        # The selected weights are also kept in last.pt so recovery is atomic.
        atomic_checkpoint(checkpoint, state["best_checkpoint"])
    atomic_json(out / "params.json", params)
    atomic_json(
        out / "training_metadata.json",
        {
            "model": name,
            "target_epochs": target_epochs,
            "seed": 42,
            "training_protocol": TRAINING_PROTOCOL,
            "mapping_digest": digest,
            "validation_digest": val_digest,
            "gradient_clip_norm": 1.0,
            "checkpoint_selection": "maximum validation nDCG@10; earliest epoch wins ties",
            "validation_batch_size": min(64, config.evaluation.batch_size),
            "smoke": smoke,
        },
    )
    best_checkpoint = None if not last_checkpoint.exists() else state["best_checkpoint"]
    end_epoch = 1 if smoke else min(target_epochs, stop_after)
    for epoch in range(start_epoch, end_epoch):
        if epoch and name == "BERT4Rec":
            loader = make_loader(model, dataset, params, epoch, generator)
        model.train()
        beginning = time.monotonic()
        total, count, steps = 0.0, 0, 0
        for batch in loader:
            batch = [tensor.to(device) for tensor in batch]
            optimizer.zero_grad(set_to_none=True)
            loss = model.train_step(batch, epoch)
            if not torch.isfinite(loss):
                raise ValueError(f"Nonfinite loss for {name} at epoch {epoch + 1}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True
            )
            optimizer.step()
            total += loss.item() * len(batch[0])
            count += len(batch[0])
            steps += 1
            if smoke and steps >= 2:
                break
        if not smoke and (count != len(loader.dataset) or steps != len(loader)):
            raise RuntimeError("Training epoch did not consume the full dataset")
        score = evaluate_validation(
            model, validation_dataset, device, min(64, config.evaluation.batch_size)
        )
        if not np.isfinite(score):
            raise ValueError("Nonfinite validation score")
        history.append(
            {
                "epoch": epoch + 1,
                "loss": total / count,
                "samples": count,
                "batches": steps,
                "validation_ndcg10": score,
                "seconds": time.monotonic() - beginning,
            }
        )
        if score > best_score:
            best_epoch, best_score = epoch + 1, score
            best_checkpoint = {
                "model_state": {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                },
                "params": params,
                "info": dataset.info(),
                "model_name": name,
                "mapping_digest": digest,
                "validation_digest": val_digest,
                "training_protocol": TRAINING_PROTOCOL,
                "epoch": best_epoch,
                "validation_ndcg10": best_score,
                "full_epoch": not smoke,
                "checkpoint_selection": "validation_ndcg10",
            }
        state = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "params": params,
            "model_name": name,
            "mapping_digest": digest,
            "validation_digest": val_digest,
            "training_protocol": TRAINING_PROTOCOL,
            "epoch": epoch + 1,
            "history": history,
            "full_epoch": not smoke,
            "best_epoch": best_epoch,
            "best_validation_ndcg10": best_score,
            "best_checkpoint": best_checkpoint,
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(),
            "loader_rng": generator.get_state(),
        }
        atomic_checkpoint(last_checkpoint, state)
        atomic_checkpoint(checkpoint, best_checkpoint)
        atomic_json(out / "training_log.json", history)
        print(
            f"{name} epoch={epoch + 1}/{target_epochs} loss={total / count:.6f} "
            f"validation_ndcg10={score:.6f} best_epoch={best_epoch}",
            flush=True,
        )
    if smoke:
        (out / "SMOKE_SUCCESS").write_text(
            "Two training batches and validation completed\n"
        )
    elif end_epoch == target_epochs:
        (out / "SUCCESS").write_text(
            f"{target_epochs} epochs completed; selected epoch {best_epoch}, "
            f"validation nDCG@10={best_score:.9f}\n"
        )
    del model, optimizer, loader
    torch.cuda.empty_cache()
    dataset.train_session._cached_dataset.clear()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--model")
    choice.add_argument("--preflight-all", action="store_true")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stop-after-epoch", type=int, default=300)
    args = parser.parse_args()
    config = load_train_configuration(args.config)
    dataset, validation, folds = initialize_datasets(
        reader=ReaderFactory.get_reader(config=config),
        callback=WarpRecCallback(),
        config=config,
    )
    if validation is None or folds:
        raise ValueError("Configure one temporal validation holdout")
    names = list(config.models) if args.preflight_all else [args.model]
    for name in names:
        output = (
            Path(args.output_dir) / name
            if args.preflight_all
            else Path(args.output_dir)
        )
        train(
            config,
            dataset,
            name,
            output,
            args.preflight_all,
            args.stop_after_epoch,
            validation_dataset=validation,
        )


if __name__ == "__main__":
    main()
