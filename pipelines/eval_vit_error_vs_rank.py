import os
import sys
import random
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from vit_pytorch import SimpleViT

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
CHECKPOINT_PATH = "../checkpoints/vit-exp/2023-01-14 23_50_08.185 dataset=cifar10 model=vit_exp epochs=200 lr_max=0.033217 model_width=512 l2_reg=0.0 sam_rho=0.05 batch_size=128 frac_train=1 p_label_noise=0.0 lr_schedule=cyclic augm=False randaug=False seed=0 epoch=200.pth"
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


# ---------------------------------------------------------------------
# Helper: load SimpleViT model
# ---------------------------------------------------------------------
def load_vit_model(checkpoint_path):
    model = SimpleViT(
        image_size=32,
        patch_size=4,
        num_classes=10,
        dim=512,
        depth=6,
        heads=16,
        mlp_dim=512 * 2,
    )

    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "last" in state:
        state = state["last"]
    model.load_state_dict(state)
    return model


# ---------------------------------------------------------------------
# Core: compute error vs rank for a single linear layer
# ---------------------------------------------------------------------
def compute_error_vs_rank_for_layer(layer_name, W_tensor, kmeans_n_init=10):
    """
    W_tensor: (out_features, in_features)
    Returns a list of dicts with pruning and folding errors for k = 1..m.
    """
    W = W_tensor.detach().cpu()
    m = W.shape[0]
    W_mat = W.view(m, -1)  # (m, d)
    original_norm2 = float(torch.sum(W_mat ** 2).item())

    # Row norms for magnitude pruning
    row_norms = torch.norm(W_mat, dim=1)
    sorted_norms, sorted_indices = torch.sort(row_norms, descending=True)

    X = W_mat.numpy()

    from sklearn.cluster import KMeans

    results = []

    for k in range(1, m + 1):
        # --- magnitude pruning with rank k ---
        keep_idx = sorted_indices[:k]
        mask = torch.zeros(m, dtype=torch.bool)
        mask[keep_idx] = True

        W_prune = torch.zeros_like(W_mat)
        W_prune[mask] = W_mat[mask]
        err_prune = float(torch.sum((W_mat - W_prune) ** 2).item())
        rel_err_prune = err_prune / original_norm2 if original_norm2 > 0 else 0.0

        # --- folding via k-means with k clusters ---
        kmeans = KMeans(
            n_clusters=k,
            n_init=kmeans_n_init,
            random_state=42,
        )
        labels = kmeans.fit_predict(X)           # (m,)
        centroids = kmeans.cluster_centers_      # (k, d)
        X_approx = centroids[labels]            # (m, d)
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


# ---------------------------------------------------------------------
# Identify FFN Linear layers in SimpleViT by name pattern
# ---------------------------------------------------------------------
def is_ffn_linear(name, module):
    """
    Selects FFN Linear layers inside the transformer blocks.

    For lucidrains' SimpleViT, typical names look like:
      - 'transformer.layers.0.0.fn.fn.to_qkv.weight'
      - 'transformer.layers.0.0.fn.fn.to_out.0.weight'
      - 'transformer.layers.0.1.fn.fn.net.0.weight'  (FFN)
      - 'transformer.layers.0.1.fn.fn.net.3.weight'  (FFN)

    We keep:
      - Linear modules whose name contains 'transformer.layers'
      - but does NOT contain 'to_qkv' or 'to_out'
    That effectively picks the FFN net.* linears.
    """
    if not isinstance(module, nn.Linear):
        return False
    if "transformer.layers" not in name:
        return False
    if "to_qkv" in name or "to_out" in name:
        return False
    return True


# ---------------------------------------------------------------------
# Main: compute error-vs-rank for each FFN layer & print CSV
# ---------------------------------------------------------------------
if __name__ == "__main__":
    model = load_vit_model(CHECKPOINT_PATH).to(DEVICE).eval()

    print("layer_name,layer_type,num_rows,k,original_norm2,err_prune,err_fold,rel_err_prune,rel_err_fold")

    for name, module in model.named_modules():
        if is_ffn_linear(name, module):
            W = module.weight
            if W is None or W.numel() == 0:
                continue

            layer_results = compute_error_vs_rank_for_layer(name, W)

            for r in layer_results:
                print(
                    f"{r['layer_name']},Linear,{r['num_rows']},"
                    f"{r['k']},{r['original_norm2']},"
                    f"{r['err_prune']},{r['err_fold']},"
                    f"{r['rel_err_prune']},{r['rel_err_fold']}"
                )
