"""Training corrections opt-in via corrected_training, preserving legacy checkpoints.

References: RuihongQiu/DuoRec (DuoRec and CL4SRec); asash/gSASRec-pytorch.
"""

import torch
from torch.nn import functional as F


def info_nce_logits(first, second, temperature, similarity="dot"):
    if temperature <= 0 or first.shape != second.shape:
        raise ValueError("Invalid contrastive inputs")
    n = len(first)
    z = torch.cat([first, second])
    if similarity == "cos":
        z = F.normalize(z, dim=-1)
    elif similarity != "dot":
        raise ValueError(similarity)
    scores = z @ z.T / temperature
    rows = torch.arange(2 * n, device=z.device)
    positive = (rows + n) % (2 * n)
    mask = torch.ones_like(scores, dtype=torch.bool)
    mask[rows, rows] = False
    mask[rows, positive] = False
    return torch.cat(
        [scores[rows, positive, None], scores[mask].reshape(2 * n, 2 * n - 2)], dim=1
    ), torch.zeros(2 * n, dtype=torch.long, device=z.device)


def calibrated_bce(logits, valid, n_items, t):
    if not 0 <= t <= 1 or n_items <= 1:
        raise ValueError("Invalid gBCE calibration")
    k = logits.shape[-1] - 1
    beta = 1 - t + t * k / (n_items - 1)
    # -log(sigmoid(pos)^beta) is exactly beta * softplus(-pos).
    loss = (
        beta * F.softplus(-logits[..., 0]) + F.softplus(logits[..., 1:]).sum(-1)
    ) / (k + 1)
    return (loss * valid).sum() / valid.sum().clamp(min=1)


def contrastive_views(
    item_seq, lengths, padding, mask_id, crop=0.6, mask=0.3, reorder=0.6
):
    """Two distinct valid-prefix augmentations, vectorized across the batch."""
    batch, width = item_seq.shape
    device = item_seq.device
    positions = torch.arange(width, device=device)[None, :].expand(batch, -1)
    first = torch.randint(3, (batch,), device=device)
    second = (first + torch.randint(1, 3, (batch,), device=device)) % 3
    views = []
    lens = []
    for choices in (first, second):
        result = item_seq.clone()
        output_lengths = lengths.clone()
        for kind in range(3):
            active = (choices == kind) & (lengths > 1)
            if not active.any():
                continue
            seq = item_seq[active]
            length = lengths[active]
            pos = positions[active]
            fraction = (crop, mask, reorder)[kind]
            count = (length * fraction).long()
            if kind == 0:
                count = count.clamp(min=1)
                start = (
                    torch.rand(len(length), device=device) * (length - count + 1)
                ).long()
                idx = (pos + start[:, None]).clamp(max=width - 1)
                result[active] = torch.where(
                    pos < count[:, None], seq.gather(1, idx), padding
                )
                output_lengths[active] = count
            elif kind == 1:
                noise = torch.rand(seq.shape, device=device).masked_fill(
                    pos >= length[:, None], torch.inf
                )
                ordering = noise.argsort(dim=1)
                ranks = torch.empty_like(ordering)
                ranks.scatter_(1, ordering, pos)
                result[active] = torch.where(ranks < count[:, None], mask_id, seq)
            else:
                start = (
                    torch.rand(len(length), device=device) * (length - count + 1)
                ).long()
                window = (pos >= start[:, None]) & (pos < (start + count)[:, None])
                noise = torch.rand(seq.shape, device=device).masked_fill(
                    ~window, torch.inf
                )
                shuffled = noise.argsort(dim=1)
                offset = (pos - start[:, None]).clamp(min=0, max=width - 1)
                permutation = shuffled.gather(1, offset)
                result[active] = torch.where(window, seq.gather(1, permutation), seq)
        views.append(result)
        lens.append(output_lengths)
    return views[0], lens[0], views[1], lens[1]
