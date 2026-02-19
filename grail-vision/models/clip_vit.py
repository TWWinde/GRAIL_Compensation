import torch
import os
import torch.nn as nn
from collections import defaultdict
from open_clip import create_model_and_transforms
from collections import OrderedDict

# === Utility functions ===
def get_module_by_name_ViT(model, module_name):
    """Traverse model by dotted name (supports list indices)."""
    parts = module_name.split('.')
    for part in parts:
        if part.isdigit():
            model = model[int(part)]
        else:
            model = getattr(model, part)
    return model


def get_axis_to_perm_ViT(model):
    """Map transformer MLP layers (c_fc and c_proj) to folding axes."""
    axis_to_perm = defaultdict(list)
    for i in range(12):
        axis_to_perm[f"group_{i}"] = [
            (f"visual.transformer.resblocks.{i}.mlp.c_fc", 0),
            (f"visual.transformer.resblocks.{i}.mlp.c_proj", 1),
        ]
    return axis_to_perm


class CLIPViT_B32:
    """
    A robust loader for CLIP ViT-B/32 with optional classification head support.
    It handles various checkpoint formats and prefix conventions automatically.
    """

    def __init__(self, checkpoint_path: str, num_classes: int = 1000, device: str = "cuda"):
        self.checkpoint_path = checkpoint_path
        self.num_classes = num_classes
        self.device = device

    def _load_ckpt(self, path):
        """Load checkpoint safely from disk."""
        if not (path and os.path.isfile(path)):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        sd = torch.load(path, map_location="cpu")
        # Some checkpoints are wrapped with {'state_dict': ...}
        if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
            sd = sd["state_dict"]
        return sd

    def _strip_prefixes(self, state_dict: dict) -> dict:
        """
        Remove common prefixes like 'module.', 'model.', or 'model.visual.'.
        This aligns checkpoint keys with model.visual state_dict().
        """
        new_sd = OrderedDict()
        for k, v in state_dict.items():
            nk = k
            if nk.startswith("module."):
                nk = nk[len("module."):]
            if nk.startswith("model.visual."):
                nk = nk[len("model.visual."):]
            elif nk.startswith("visual."):
                nk = nk[len("visual."):]
            new_sd[nk] = v
        return new_sd

    def load(self):
        """
        Build a CLIP ViT-B/32 model using open_clip, attach classification head,
        and load visual + head weights from checkpoint.
        """
        model, _, preprocess = create_model_and_transforms(
            model_name="ViT-B-32",
            pretrained="",        # do not load any default weights
            precision="fp32"
        )
        model = model.to(self.device)

        # Attach a linear classification head (for ImageNet etc.)
        model.classification_head = nn.Linear(model.visual.output_dim, self.num_classes).to(self.device)

        # Load the checkpoint
        raw = self._load_ckpt(self.checkpoint_path)

        # --- Load classification head (if present) ---
        head_sd = OrderedDict(
            (k[len("classification_head."):], v)
            for k, v in raw.items()
            if k.startswith("classification_head.")
        )
        if head_sd:
            miss_h, unexp_h = model.classification_head.load_state_dict(head_sd, strict=False)
            print(f"[head] missing={len(miss_h)}, unexpected={len(unexp_h)}")

        # --- Load vision branch ---
        vis_raw = {k: v for k, v in raw.items() if k.startswith("model.visual.") or k.startswith("visual.")}
        vis_sd = self._strip_prefixes(vis_raw)

        miss_v, unexp_v = model.visual.load_state_dict(vis_sd, strict=False)
        print(f"[visual] missing={len(miss_v)}, unexpected={len(unexp_v)}")
        if miss_v[:10]:
            print("  missing (first 10):", miss_v[:10])
        if unexp_v[:10]:
            print("  unexpected (first 10):", unexp_v[:10])

        model.eval()
        return model, preprocess









