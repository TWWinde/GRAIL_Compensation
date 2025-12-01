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
# Not strictly needed here, but we keep the structure similar to your script:
# from compression.fold import ResNet18_ModelFolding
# from compression.mag_prune import ResNet18_MagnitudePruning

# -------------------------------------------------------------------------
# Config
# -------------------------------------------------------------------------
CHECKPOINT_PATH = "../checkpoints/resnet18/adam/2025-06-08_08-18-22_dataset=cifar10_arch=resnet18_opt=adam_seed=42_lr=0.01_batch_size=128_momentum=0.0_wd=0.0_epochs=200_l1=1e-05_l2=0.0_sam=False_sam_rho=0.05_rand_aug=False.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
        print(f"[INFO] Loading checkpoint: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict)
    else:
        print("[WARNING] No checkpoint found, using randomly initialized model.")
    return model


# -------------------------------------------------------------------------
# Core experiment: error vs. rank for each layer
# -------------------------------------------------------------------------
def compute_error_vs_rank_for_layer(layer_name, W_tensor, kmeans_n_init=10):
    """
    Given a weight tensor W_tensor of shape:
      - Conv2d: (out_channels, in_channels, kH, kW)
      - Linear: (out_features, in_features)
    compute squared Frobenius reconstruction errors for:
      - magnitude pruning (axis-aligned)
      - folding via k-means (row clustering)
    for all ranks k = 1, ..., m (m = #rows).
    Returns a list of dicts, one per k.
    """
    # Move to CPU and flatten to (m, d)
    W = W_tensor.detach().cpu()
    m = W.shape[0]
    W_mat = W.view(m, -1)  # (m, d)
    original_norm2 = float(torch.sum(W_mat ** 2).item())

    # Precompute row norms and ordering for pruning
    row_norms = torch.norm(W_mat, dim=1)  # (m,)
    # Sort descending; we'll use prefixes for each k
    sorted_norms, sorted_indices = torch.sort(row_norms, descending=True)

    # Prepare data for k-means (numpy)
    X = W_mat.numpy()

    from sklearn.cluster import KMeans

    results = []

    for k in range(1, m + 1):
        # -------------------------------
        # Magnitude pruning with rank k
        # -------------------------------
        keep_idx = sorted_indices[:k]
        mask = torch.zeros(m, dtype=torch.bool)
        mask[keep_idx] = True

        W_prune = torch.zeros_like(W_mat)
        W_prune[mask] = W_mat[mask]
        err_prune = float(torch.sum((W_mat - W_prune) ** 2).item())
        rel_err_prune = err_prune / original_norm2 if original_norm2 > 0 else 0.0

        # -------------------------------
        # Folding via k-means with k clusters
        # -------------------------------
        # Note: this is independent clustering per k
        # (simple, but potentially expensive – OK for offline analysis)
        kmeans = KMeans(
            n_clusters=k,
            n_init=kmeans_n_init,
            random_state=42,
        )
        labels = kmeans.fit_predict(X)              # (m,)
        centroids = kmeans.cluster_centers_         # (k, d)
        X_approx = centroids[labels]                # (m, d)
        W_fold = torch.from_numpy(X_approx).float()
        err_fold = float(torch.sum((W_mat - W_fold) ** 2).item())
        rel_err_fold = err_fold / original_norm2 if original_norm2 > 0 else 0.0

        results.append(
            dict(
                layer_name=layer_name,
                num_rows=m,
                k=k,
                original_norm2=original_norm2,
                err_prune=err_prune,
                err_fold=err_fold,
                rel_err_prune=rel_err_prune,
                rel_err_fold=rel_err_fold,
            )
        )

    return results


def main():
    num_classes = 10
    model = load_resnet18_model(num_classes, CHECKPOINT_PATH).to(DEVICE)
    model.eval()

    # We'll process only Conv2d and Linear layers
    print("layer_name,layer_type,num_rows,k,original_norm2,err_prune,err_fold,rel_err_prune,rel_err_fold")

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            layer_type = module.__class__.__name__
            W = module.weight

            # Skip bias-only / degenerate layers just in case
            if W is None or W.numel() == 0:
                continue

            layer_results = compute_error_vs_rank_for_layer(name, W)

            for r in layer_results:
                print(
                    f"{r['layer_name']},{layer_type},{r['num_rows']},"
                    f"{r['k']},{r['original_norm2']},"
                    f"{r['err_prune']},{r['err_fold']},"
                    f"{r['rel_err_prune']},{r['rel_err_fold']}"
                )


if __name__ == "__main__":
    main()
