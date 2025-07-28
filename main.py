import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageNet

from models.clip import CLIPViT_B32
from architectures.clip_vit import CLIPViT_ModelFolding
from compression.utils.eval_utils import test, count_parameters  # move helper funcs here


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    # === 1. Load CLIP ViT-B/32 model with model-soups checkpoint ===
    checkpoint_path = "../checkpoints/clipvit-b32-model-soups/model_0.pt"
    num_classes = 1000

    model, preprocess = CLIPViT_B32(
        checkpoint_path=checkpoint_path,
        num_classes=num_classes,
        device=device
    )

    # === 2. Prepare ImageNet validation loader ===
    val_dataset = ImageNet(root="../data", split="val", transform=preprocess)
    val_loader = DataLoader(
        val_dataset, batch_size=32, shuffle=False,
        num_workers=4, pin_memory=True
    )

    # === 3. Evaluate BEFORE folding ===
    print("\n=== Evaluation BEFORE folding ===")
    original_params = count_parameters(model)
    acc_before = test(model, val_loader, device=device)
    print(f"Top-1 Accuracy: {acc_before:.2f}%")
    print(f"Original Parameters: {original_params}")

    # === 4. Apply Folding ===
    folding = CLIPViT_ModelFolding(model, normalize=True, compression_ratio=0.5)
    folded_model = folding.apply()

    # === 5. Evaluate AFTER folding ===
    print("\n=== Evaluation AFTER folding ===")
    pruned_params = count_parameters(folded_model)
    acc_after = test(folded_model, val_loader, device=device)
    print(f"Top-1 Accuracy: {acc_after:.2f}%")
    print(f"Pruned Parameters: {pruned_params}")
    print(f"Compression Ratio: {(original_params - pruned_params) / original_params:.2%}")


if __name__ == "__main__":
    main()

