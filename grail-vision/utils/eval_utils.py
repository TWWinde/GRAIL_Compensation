import torch
import torch.nn.functional as F

# --- Count parameters
def count_parameters(model):
    """
    Count total trainable parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# --- Test function (Top-1 Accuracy)
def test(model, test_loader, device):
    """
    Evaluate model accuracy on test set.
    - ResNet: model(images) returns logits
    - CLIP ViT: must use model.visual(images) + classification_head
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            # --- Forward pass ---
            if hasattr(model, "classification_head") and hasattr(model, "visual"):
                # CLIP ViT path
                outputs = model.classification_head(model.visual(images))
            else:
                # ResNet path
                outputs = model(images)

            # --- Compute predictions ---
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return 100.0 * correct / total


# --- Get outputs from the model
def get_outputs(model, loader, device):
    outputs = []
    with torch.no_grad():
        for x, _ in loader:
            if hasattr(model, "classification_head") and hasattr(model, "visual"):
                out = model.classification_head(model.visual(x.to(device)))
            else:
                out = model(x.to(device))
            outputs.append(out.cpu())
    return torch.cat(outputs, dim=0)


# --- Compute functional deviation
def functional_deviation(orig_outputs, new_outputs):
    """
    Compute mean L2 distance between original and new outputs.
    """
    return torch.norm(orig_outputs - new_outputs, dim=1).mean().item()
