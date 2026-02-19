import os
import sys
import random
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageNet

from models.clip_vit import CLIPViT_B32

from compression.fold import CLIPViT_ModelFolding
from compression.mag_prune import CLIPViT_MagnitudePruning
from compression.wanda_prune import CLIPViT_WandaPruning
from compression.rand_fold import CLIPViT_RandomFolding
from compression.rand_prune import CLIPViT_RandomPruning
#from compression.singleton import CLIPViT_Singleton
from compensation.prune_compensation import CLIPViT_PruneCompensation
from compensation.folding_compensation import CLIPViT_FoldingCompensation

from utils.tune_utils import retune_layernorm
from utils.eval_utils import test, count_parameters

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
CHECKPOINT_PATH = "./checkpoints/clipvit-b32-model-soups/model_11.pt"
BATCH_SIZE = 32
COMPRESSION_RATIO = 0.2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODE = "fold"  # options: "fold" or "prune"

def fix_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
fix_seed(42)

# -----------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------
def load_clip_vit_model(num_classes, checkpoint_path, device):
    """
    Load CLIP ViT-B/32 model and preprocessing transform.
    """
    clip_loader = CLIPViT_B32(
        checkpoint_path=checkpoint_path,
        num_classes=num_classes,
        device=device
    )

    return clip_loader.load()  # returns (model, preprocess)

# -----------------------------------------------------------------------------
# Main folding evaluation
# -----------------------------------------------------------------------------
def main():
    num_classes = 1000  # ImageNet classes
    model, preprocess = load_clip_vit_model(num_classes, CHECKPOINT_PATH, DEVICE)

    # Prepare validation loader
    val_dataset = ImageNet(root="/gpfs/data/fs72923/tang/projects/data", split="val", transform=preprocess)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    # Evaluate BEFORE folding
    print("\n=== Evaluation BEFORE compression ===")
    acc_before = test(model, val_loader, device=DEVICE)
    print(f"🔹 Top-1 Accuracy: {acc_before:.2f}%")
    original_params = count_parameters(model)
    print(f"Original Parameters: {original_params}")

    print("\n[INFO] Applying CLIP ViT-B/32 model compression...")
    if MODE == "fold":
        folder = CLIPViT_ModelFolding(model, compression_ratio=COMPRESSION_RATIO)
        folder.run_calibration(val_loader, DEVICE, num_batches=100, forward_fn=model.encode_image)
        folded_model = folder.apply()

        fold_stats = folder.get_compression_state()
        gram_stats = folder.get_gram_stats()

        print("\n=== Evaluation AFTER folding (before compensation) ===")
        folded_params = count_parameters(folded_model)
        acc_after = test(folded_model, val_loader, device=DEVICE)
        print(f"🔹 Top-1 Accuracy: {acc_after:.2f}%")
        print(f"Folded Parameters: {folded_params}")

        compensator = CLIPViT_FoldingCompensation(folded_model, ridge_lambda=1e-3)
        compensator.load_compression_state(fold_stats)
        compensator.load_gram_stats(gram_stats)
        compensated_model = compensator.apply()

        print("\n=== Evaluation AFTER folding + compensation (before LN tune) ===")
        compensated_params = count_parameters(compensated_model)
        acc_after_comp = test(compensated_model, val_loader, device=DEVICE)
        print(f"🔹 Top-1 Accuracy: {acc_after_comp:.2f}%")

        print("\n[INFO] Re-tuning LayerNorm parameters...")
        retune_layernorm(compensated_model, val_loader, device=DEVICE, lr=1e-4)
        acc_after_ln = test(compensated_model, val_loader, device=DEVICE)
        print(f"🔹 Top-1 Accuracy (after LN re-tune): {acc_after_ln:.2f}%")
        print(f"🔥 Compression Ratio: {(original_params - compensated_params) / original_params:.2%}")
    else:
        pruner = CLIPViT_WandaPruning(model, compression_ratio=COMPRESSION_RATIO, mode="proj_cols", ignore_cls=True, alpha=0.35)
        pruner.run_calibration(val_loader, DEVICE, num_batches=100, forward_fn=model.encode_image)
        pruned_model = pruner.apply()

        print("\n=== Evaluation AFTER compression (before LN re-tune) ===")
        pruned_params = count_parameters(pruned_model)
        acc_after = test(pruned_model, val_loader, device=DEVICE)
        print(f"🔹 Top-1 Accuracy: {acc_after:.2f}%")

        prune_entries = pruner.get_prune_state()
        gram_stats = pruner.get_gram_stats()

        compensator = CLIPViT_PruneCompensation(pruned_model, ridge_lambda=1e-3)
        compensator.load_prune_state(prune_entries)
        compensator.load_gram_stats(gram_stats)
        compensated_model = compensator.apply()

        print("\n=== Evaluation AFTER compression and compensation (before REPAIR) ===")
        compensated_params = count_parameters(compensated_model)
        acc_after_comp = test(compensated_model, val_loader, device=DEVICE)
        print(f"🔹 Top-1 Accuracy: {acc_after_comp:.2f}%")

        print("\n[INFO] Re-tuning LayerNorm parameters...")
        retune_layernorm(pruned_model, val_loader, device=DEVICE, lr=1e-4)

        print("\n=== Evaluation AFTER compression compensation and repair===")
        acc_after_ln = test(pruned_model, val_loader, device=DEVICE)
        print(f"🔹 Top-1 Accuracy (after LN re-tune): {acc_after_ln:.2f}%")
        print(f"Pruned Parameters: {pruned_params}")
        print(f"🔥 Compression Ratio: {(original_params - pruned_params) / original_params:.2%}")

if __name__ == "__main__":
    main()
