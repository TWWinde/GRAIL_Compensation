import os
import sys
import random
import numpy as np

import torch
import torch.nn as nn

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# -------------------------------------------------------------------------
# Repo imports (adjust path if needed)
# -------------------------------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.resnet import ResNet18

# -------------------------------------------------------------------------
# Config
# -------------------------------------------------------------------------
CHECKPOINT_PATH = "../checkpoints/resnet18/adam/2025-06-08_08-18-22_dataset=cifar10_arch=resnet18_opt=adam_seed=42_lr=0.01_batch_size=128_momentum=0.0_wd=0.0_epochs=200_l1=1e-05_l2=0.0_sam=False_sam_rho=0.05_rand_aug=False.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# pruning levels (k_p = round(keep_ratio * m))
KEEP_RATIOS = [0.25, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

def fix_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

fix_seed(42)

# -------------------------------------------------------------------------
# Model loading
# -------------------------------------------------------------------------
def load_resnet18_model(num_classes, checkpoint_path=None):
    model = ResNet18(num_classes=num_classes)
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"[INFO] Loading checkpoint: {checkpoint_path}", file=sys.stderr)
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict)
    else:
        print("[WARNING] No checkpoint found, using randomly initialized model.", file=sys.stderr)
    return model


# -------------------------------------------------------------------------
# Helpers for 5.2 experiment
# -------------------------------------------------------------------------
def flatten_rows(W: torch.Tensor) -> torch.Tensor:
    """
    Conv2d: (out_c, in_c, kH, kW) -> (out_c, in_c * kH * kW)
    Linear: (out_f, in_f) -> (out_f, in_f)
    """
    return W.view(W.size(0), -1)


def magnitude_prune_rows(W: torch.Tensor, keep_ratio: float):
    """
    Structured magnitude pruning on rows (output channels).
    Returns:
        W_p: pruned weights (same shape as W)
        keep_idx: indices of kept rows
        prune_idx: indices of pruned rows
    """
    W_flat = flatten_rows(W)
    m = W_flat.size(0)
    k_p = max(1, int(round(m * keep_ratio)))
    k_p = min(k_p, m)

    norms = torch.norm(W_flat, dim=1)
    keep_idx = torch.topk(norms, k=k_p, largest=True).indices
    mask = torch.zeros(m, dtype=torch.bool, device=W.device)
    mask[keep_idx] = True
    prune_idx = (~mask).nonzero(as_tuple=False).view(-1)

    W_p_flat = W_flat.clone()
    W_p_flat[prune_idx] = 0.0
    W_p = W_p_flat.view_as(W)

    return W_p, keep_idx.cpu(), prune_idx.cpu(), k_p


def special_folding_from_pruning(W: torch.Tensor,
                                 keep_idx: torch.Tensor,
                                 prune_idx: torch.Tensor):
    """
    Special folding W_f' from the proof:
      - kept rows unchanged;
      - all pruned rows replaced by their mean.
    """
    W_flat = flatten_rows(W).detach().cpu()
    if prune_idx.numel() == 0:
        return W.clone()  # nothing pruned

    mu = W_flat[prune_idx].mean(dim=0, keepdim=True)  # (1, d)
    W_f_flat = W_flat.clone()
    W_f_flat[prune_idx] = mu
    return W_f_flat.view_as(W)


def kmeans_folding(W: torch.Tensor,
                   num_clusters: int,
                   n_init: int = 10,
                   random_state: int = 42):
    """
    Optimal folding W_f^* via k-means clustering on rows.
    """
    from sklearn.cluster import KMeans

    W_flat = flatten_rows(W).detach().cpu()
    m, d = W_flat.shape
    num_clusters = min(num_clusters, m)

    X = W_flat.numpy()
    kmeans = KMeans(
        n_clusters=num_clusters,
        n_init=n_init,
        random_state=random_state,
    )
    labels = kmeans.fit_predict(X)
    centroids = kmeans.cluster_centers_  # (num_clusters, d)

    X_approx = centroids[labels]        # (m, d)
    W_f_flat = torch.from_numpy(X_approx).float()
    return W_f_flat.view_as(W)


def compute_5_2_for_layer(layer_name, layer_type, W_tensor, keep_ratios):
    """
    Implements 5.2 for a single layer:
      - W_p: magnitude pruning with k_p rows
      - W_f': special folding (merge pruned rows into one cluster)
      - W_f^*: k-means optimal folding with k_f = k_p + 1 clusters

    Returns a list of dicts (one per keep_ratio).
    """
    W = W_tensor.detach().cpu()
    m = W.shape[0]
    W_flat = flatten_rows(W)
    original_norm2 = float(torch.sum(W_flat ** 2).item())

    results = []

    for keep_ratio in keep_ratios:
        # pruning
        W_p, keep_idx, prune_idx, k_p = magnitude_prune_rows(W, keep_ratio)

        # special folding W_f'
        W_f_special = special_folding_from_pruning(W, keep_idx, prune_idx)

        # k-means folding W_f^*
        k_f = k_p + 1 if k_p < m else k_p
        W_f_opt = kmeans_folding(W, num_clusters=k_f)

        # errors
        err_p = float(torch.sum((W - W_p) ** 2).item())
        err_f_sp = float(torch.sum((W - W_f_special) ** 2).item())
        err_f_opt = float(torch.sum((W - W_f_opt) ** 2).item())

        rel_err_p = err_p / original_norm2 if original_norm2 > 0 else 0.0
        rel_err_f_sp = err_f_sp / original_norm2 if original_norm2 > 0 else 0.0
        rel_err_f_opt = err_f_opt / original_norm2 if original_norm2 > 0 else 0.0

        results.append(
            dict(
                layer_name=layer_name,
                layer_type=layer_type,
                num_rows=m,
                keep_ratio=float(keep_ratio),
                k_p=int(k_p),
                k_f=int(k_f),
                original_norm2=original_norm2,
                err_prune=err_p,
                err_special_fold=err_f_sp,
                err_opt_fold=err_f_opt,
                rel_err_prune=rel_err_p,
                rel_err_special_fold=rel_err_f_sp,
                rel_err_opt_fold=rel_err_f_opt,
            )
        )

    return results


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
def main():
    num_classes = 10
    model = load_resnet18_model(num_classes, CHECKPOINT_PATH).to(DEVICE)
    model.eval()

    # CSV header to stdout
    print(
        "layer_name,layer_type,num_rows,keep_ratio,k_p,k_f,"
        "original_norm2,err_prune,err_special_fold,err_opt_fold,"
        "rel_err_prune,rel_err_special_fold,rel_err_opt_fold"
    )

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            layer_type = module.__class__.__name__
            W = module.weight
            if W is None or W.numel() == 0:
                continue

            layer_results = compute_5_2_for_layer(
                layer_name=name,
                layer_type=layer_type,
                W_tensor=W,
                keep_ratios=KEEP_RATIOS,
            )

            for r in layer_results:
                print(
                    f"{r['layer_name']},{r['layer_type']},{r['num_rows']},"
                    f"{r['keep_ratio']},{r['k_p']},{r['k_f']},"
                    f"{r['original_norm2']},"
                    f"{r['err_prune']},{r['err_special_fold']},{r['err_opt_fold']},"
                    f"{r['rel_err_prune']},{r['rel_err_special_fold']},{r['rel_err_opt_fold']}"
                )


if __name__ == "__main__":
    main()
