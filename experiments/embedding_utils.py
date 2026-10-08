"""Embedding extraction and distance calculation."""
import numpy as np
def distance_matrix(x, y, metric):
    if metric == "cosine":
        nx, ny = np.linalg.norm(x, axis=1), np.linalg.norm(y, axis=1)
        if np.any(nx <= 1e-12) or np.any(ny <= 1e-12):
            raise ValueError("Cosine distance undefined for a zero embedding/centroid; use euclidean")
        return np.clip(1 - (x / nx[:, None]) @ (y / ny[:, None]).T, 0, 2)
    # Compiled implementation avoids a Python loop for long full-history profiles.
    from scipy.spatial.distance import cdist
    return cdist(x, y, metric="euclidean")


def extract_embeddings(model, item_ids, device):
    import torch

    layer = getattr(model, "item_embedding", None)
    if layer is None:
        layer = getattr(model, "item_codes_layer", None)
    if layer is None:
        raise TypeError(f"{type(model).__name__}: no supported input item embedding layer")
    with torch.inference_mode():
        vectors = layer(torch.tensor(item_ids, dtype=torch.long, device=device))
    return vectors.detach().cpu().numpy().astype(np.float64)

