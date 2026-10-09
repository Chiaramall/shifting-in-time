"""Embedding extraction and distance calculation."""

import numpy as np


def extract_embeddings(model, item_ids, device):
    import torch

    layer = getattr(model, "item_embedding", None)
    if layer is None:
        layer = getattr(model, "item_codes_layer", None)
    if layer is None:
        raise TypeError(
            f"{type(model).__name__}: no supported input item embedding layer"
        )
    with torch.inference_mode():
        vectors = layer(torch.tensor(item_ids, dtype=torch.long, device=device))
    return vectors.detach().cpu().numpy().astype(np.float64)
