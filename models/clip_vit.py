import torch
import torch.nn as nn
from collections import defaultdict
from open_clip import create_model_and_transforms


# --- Utility functions ===
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
    axis_to_perm = defaultdict(list)
    for i in range(12):
        axis_to_perm[f"group_{i}"] = [
            (f"visual.transformer.resblocks.{i}.mlp.c_fc", 0),
            (f"visual.transformer.resblocks.{i}.mlp.c_proj", 1),
        ]
    return axis_to_perm


class CLIPViT_B32:
    def __init__(self, checkpoint_path: str, num_classes: int = 1000, device: str = "cuda"):
        self.checkpoint_path = checkpoint_path
        self.num_classes = num_classes
        self.device = device

    def load(self):
        model, _, preprocess = create_model_and_transforms(
            model_name="ViT-B-32",
            pretrained="",  # no weights loaded here
            precision="fp32"
        )
        model = model.to(self.device)

        # --- Attach classification head ===
        model.classification_head = torch.nn.Linear(model.visual.output_dim, self.num_classes).to(self.device)

        # --- Load vision + head weights ===
        raw_ckpt = torch.load(self.checkpoint_path, map_location="cpu")
        state_dict = {}
        for k, v in raw_ckpt.items():
            if k.startswith("model.visual."):
                state_dict[k.replace("model.visual.", "")] = v
            elif k.startswith("classification_head."):
                model.classification_head.state_dict()[k.replace("classification_head.", "")].copy_(v)

        missing, unexpected = model.visual.load_state_dict(state_dict, strict=False)
        print("✅ Vision + classification head loaded")

        return model, preprocess









