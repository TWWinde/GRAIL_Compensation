import torch
import torch.nn as nn
from collections import defaultdict
from models.vit import get_axis_to_perm_ViT_Simple, get_module_by_name_ViT_Simple
from utils.weight_clustering import WeightClustering


class BaseViTCompression:
    """
    Base class for standard ViT compression (folding or pruning).
    - Handles parameter extraction and replacement of nn.Linear layers
    - Assumes model structure similar to SimpleViT (not CLIP)
    """

    def __init__(self, model, min_channels=1, compression_ratio=0.5):
        self.model = model
        self.min_channels = min_channels
        self.keep_ratio = 1.0 - compression_ratio
        self.device = next(model.parameters()).device

    # --- To be implemented by subclasses
    def compress_function(self, axes, params):
        raise NotImplementedError

    # --- Helper: rebuild Linear layers
    def _rebuild_linear(self, param_dict, in_features, out_features):
        linear = nn.Linear(in_features, out_features, bias='bias' in param_dict).to(self.device)
        linear.weight.data.copy_(param_dict['weight'])
        if 'bias' in param_dict:
            linear.bias.data.copy_(param_dict['bias'])
        return linear

    # --- Main apply() logic
    def apply(self):
        print(f"[INFO] Starting {self.__class__.__name__} compression...")
        axis_to_perm = get_axis_to_perm_ViT_Simple(self.model)

        for group_name, (module_name_fc, module_name_proj) in axis_to_perm.items():
            # --- Get modules ---
            module_fc = get_module_by_name_ViT_Simple(self.model, module_name_fc)
            module_proj = get_module_by_name_ViT_Simple(self.model, module_name_proj)

            W_fc = module_fc.weight.data
            W_proj = module_proj.weight.data

            # --- Determine number of clusters ---
            target_channels = W_fc.shape[0]
            n_clusters = max(int(target_channels * self.keep_ratio), self.min_channels)
            if n_clusters >= target_channels:
                print(f"[WARNING] Skipping folding for {group_name}: "
                      f"n_clusters={n_clusters} >= target_channels={target_channels}")
                continue

            # --- Prepare raw params dictionary ---
            raw_params = {
                module_name_fc + '.weight': W_fc,
                module_name_fc + '.bias': module_fc.bias.data if module_fc.bias is not None else None,
                module_name_proj + '.weight': W_proj,
                module_name_proj + '.bias': module_proj.bias.data if module_proj.bias is not None else None,
            }

            # --- Call compress_function with BOTH fc and proj paths ---
            compressed_params, _ = self.compress_function(
                (module_name_fc, module_name_proj),  # pass tuple
                raw_params
            )

            # --- Build new fc layer ---
            new_fc = nn.Linear(
                in_features=compressed_params[module_name_fc + '.weight'].shape[1],
                out_features=compressed_params[module_name_fc + '.weight'].shape[0],
                bias=module_fc.bias is not None
            ).to(self.device)
            new_fc.weight.data.copy_(compressed_params[module_name_fc + '.weight'])
            if module_fc.bias is not None and compressed_params.get(module_name_fc + '.bias') is not None:
                new_fc.bias.data.copy_(compressed_params[module_name_fc + '.bias'])

            # --- Build new proj layer ---
            new_proj = nn.Linear(
                in_features=compressed_params[module_name_proj + '.weight'].shape[1],
                out_features=compressed_params[module_name_proj + '.weight'].shape[0],
                bias=module_proj.bias is not None
            ).to(self.device)
            new_proj.weight.data.copy_(compressed_params[module_name_proj + '.weight'])
            if module_proj.bias is not None and compressed_params.get(module_name_proj + '.bias') is not None:
                new_proj.bias.data.copy_(compressed_params[module_name_proj + '.bias'])

            # --- Replace in parent modules ---
            parent_fc = get_module_by_name_ViT_Simple(self.model, '.'.join(module_name_fc.split('.')[:-1]))
            parent_proj = get_module_by_name_ViT_Simple(self.model, '.'.join(module_name_proj.split('.')[:-1]))
            setattr(parent_fc, module_name_fc.split('.')[-1], new_fc)
            setattr(parent_proj, module_name_proj.split('.')[-1], new_proj)

        return self.model


