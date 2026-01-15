from collections import defaultdict

import numpy as np
import torch
from torch import Tensor


def get_signature_size(d: int, depth: int) -> int:
    return d + sum([d**i for i in range(2, depth + 1)])


def compute_gae(
    rewards: Tensor,
    values: Tensor,
    gamma: float = 0.99,
    lam: float = 0.95
) -> Tensor:
    advantages = []
    gae = 0
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * values[i+1] - values[i]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
    return advantages


def compute_entropy(x: Tensor) -> Tensor:
    if len(x) == 0:
        return torch.tensor(1.0, device=x.device)

    p = torch.bincount(torch.unique(x, return_inverse=True)[1])
    p = p[p > 0]

    if p.size() == 1:
        return torch.tensor(0.0, device=x.device)

    n = p.sum()
    entropy = -torch.sum((p / n) * (torch.log(p) - torch.log(n)))
    return entropy


def compute_contingency_matrix(preds: Tensor, target: Tensor) -> Tensor:
    preds_classes, preds_idx = torch.unique(preds, return_inverse=True)
    target_classes, target_idx = torch.unique(target, return_inverse=True)

    num_classes_preds = preds_classes.size(0)
    num_classes_target = target_classes.size(0)

    contingency = torch.sparse_coo_tensor(
        torch.stack((target_idx, preds_idx)),
        torch.ones(target_idx.shape[0], dtype=preds_idx.dtype, device=preds_idx.device),
        (num_classes_target, num_classes_preds)
    )

    contingency = contingency.to_dense()
    return contingency


def compute_mutual_info(predicted: Tensor, gt: Tensor) -> Tensor:
    contingency = compute_contingency_matrix(predicted, gt)
    n = contingency.sum()
    u = contingency.sum(dim=1)
    v = contingency.sum(dim=0)

    if u.size() == 1 or v.size() == 1:
        return torch.tensor(0.0)

    nzu, nzv = torch.nonzero(contingency, as_tuple=True)
    contingency = contingency[nzu, nzv]

    log_outer = torch.log(u[nzu]) + torch.log(v[nzv])
    mutual_info = contingency / n * (torch.log(n) + torch.log(contingency) - log_outer)
    return mutual_info.sum()


def clustering_scores(predicted: Tensor, gt: Tensor) -> Tensor:
    if len(predicted.size()) > 1:
        predicted = torch.argmax(predicted, 1)

    entropy_pred = compute_entropy(predicted)
    entropy_gt = compute_entropy(gt)
    mutual_info = compute_mutual_info(predicted, gt)

    completeness = mutual_info / entropy_pred if entropy_pred else torch.ones_like(entropy_pred)
    homogeneity = mutual_info / entropy_gt if entropy_gt else torch.ones_like(entropy_gt)
    return completeness.detach().cpu().item(), homogeneity.detach().cpu().item()


def get_clusters(
    predicted: Tensor,
    ground_truth: Tensor,
    _pred: dict[list[int]] = None,
    out: bool = False
) -> tuple[dict[list[int]]]:
    if not out:
        _pred = defaultdict(list)

    for pred, gt in zip(predicted, ground_truth):
        _pred[pred.argmax(dim=0).item()].append(gt.item())

    if not out:
        return _pred


def get_clustering_metrics(logits: dict[list[int]], n_classes: int) -> tuple[float]:
    homogenity = {}
    purity = {}
    for key, value in logits.items():
        count = np.bincount(value, minlength=n_classes)
        homogenity[int(key)] = count[int(key)] / count.sum()
        purity[int(key)] = count[np.argmax(count)] / count.sum()
    return np.mean(list(purity.values())), np.mean(list(homogenity.values()))


def error(predicted: Tensor, gt: Tensor, p: int = 1, reduce: str = "mean") -> Tensor:
    error = torch.nn.PairwiseDistance(p)

    if predicted.device != gt.device:
        predicted = predicted.detach().cpu()
        gt = gt.detach().cpu()

    if reduce == "mean":
        return error(predicted, gt).mean().item()
    else:
        return error(predicted, gt).sum().item()
