import torch
import torch.nn as nn
from collections import defaultdict
from models.clip_vit import get_axis_to_perm_ViT, get_module_by_name_ViT


class BaseCLIPViTCompression:
    """
    Base class for CLIP ViT compression (folding or pruning).
    Handles parameter extraction, rebuilding nn.Linear layers,
    and replacing modules in the model.
    """

    def __init__(self, model, min_channels=1, compression_ratio=0.5):
        self.model = model
        self.min_channels = min_channels
        self.keep_ratio = 1.0 - compression_ratio  # same logic as original
        self.device = next(model.parameters()).device

    def compress_function(self, axes, params):
        """
        To be implemented by subclasses:
        Should return (compressed_params, merge_sizes)
        """
        raise NotImplementedError

    def _rebuild_linear(self, param_dict, in_features, out_features):
        linear = nn.Linear(in_features, out_features, bias='bias' in param_dict).to(self.device)
        linear.weight.data.copy_(param_dict['weight'])
        if 'bias' in param_dict:
            linear.bias.data.copy_(param_dict['bias'])
        return linear

    def apply(self):
        print(f"[INFO] Starting {self.__class__.__name__} compression...")
        axis_to_perm = get_axis_to_perm_ViT(self.model)
        embed_dim = self.model.visual.conv1.out_channels  # fixed output dim

        for group_name, axes in axis_to_perm.items():
            # --- Gather parameters ---
            module_fc = get_module_by_name_ViT(self.model, axes[0][0])
            module_proj = get_module_by_name_ViT(self.model, axes[1][0])

            raw_params = {
                axes[0][0]: module_fc.weight.data,
                axes[0][0] + '.bias': module_fc.bias.data if module_fc.bias is not None else None,
                axes[1][0]: module_proj.weight.data,
                axes[1][0] + '.bias': module_proj.bias.data if module_proj.bias is not None else None,
            }

            # --- Perform compression (folding or pruning) ---
            compressed_params, _ = self.compress_function(axes, raw_params)

            # --- Rebuild and replace layers ---
            param_groups = defaultdict(dict)
            for full_name, tensor in compressed_params.items():
                module_name, pname = full_name.rsplit('.', 1)
                param_groups[module_name][pname] = tensor

            for module_name, param_dict in param_groups.items():
                module = get_module_by_name_ViT(self.model, module_name)

                # Infer shape
                if module_name.endswith('c_fc'):
                    out_features, in_features = param_dict['weight'].shape
                elif module_name.endswith('c_proj'):
                    out_features, in_features = embed_dim, param_dict['weight'].shape[1]
                else:
                    out_features, in_features = module.out_features, module.in_features

                new_linear = self._rebuild_linear(param_dict, in_features, out_features)

                # Replace in model
                parent_name = '.'.join(module_name.split('.')[:-1])
                attr_name = module_name.split('.')[-1]
                parent = get_module_by_name_ViT(self.model, parent_name) if parent_name else self.model
                setattr(parent, attr_name, new_linear)

        return self.model
