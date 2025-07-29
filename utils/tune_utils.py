import torch
import torch.nn as nn

def retune_layernorm(model, dataloader, device='cuda', lr=1e-4, steps=None):
    """
    Fine-tunes only LayerNorm parameters to stabilize accuracy after compression.
    Works for both CLIP ViT and SimpleViT (detects by module type, not name).

    Args:
        model: Vision Transformer model (CLIP or SimpleViT).
        dataloader: DataLoader for validation or train split.
        device: Device for training.
        lr: Learning rate for LN tuning.
        steps: Number of steps (None = 1 full pass over dataloader).
    """

    # --- Collect LayerNorm parameters by module type ---
    ln_params = [
        p for m in model.modules() if isinstance(m, nn.LayerNorm)
        for p in m.parameters() if p.requires_grad
    ]

    if not ln_params:
        print("[WARNING] No LayerNorm parameters found to tune.")
        return

    optimizer = torch.optim.Adam(ln_params, lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    step_count = 0

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        # --- Forward pass ---
        if hasattr(model, "visual") and hasattr(model, "classification_head"):
            # CLIP-style
            visual_output = model.visual(inputs)
            outputs = model.classification_head(visual_output)
        else:
            # SimpleViT-style
            outputs = model(inputs)

        # --- Validate output shape and targets ---
        if outputs.ndim != 2:
            raise ValueError(f"Unexpected output shape: {outputs.shape}")

        if targets.min() < 0 or targets.max() >= outputs.size(1):
            raise ValueError(f"Invalid target values: min={targets.min().item()}, max={targets.max().item()}")

        # --- Backpropagation ---
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        step_count += 1
        if steps and step_count >= steps:
            break

    model.eval()


def repair_bn(model, dataloader, device='cuda'):
    """
    Fine-tunes BatchNorm parameters using unlabeled data to stabilize accuracy after compression.
    """
    model.train()
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            model(inputs)