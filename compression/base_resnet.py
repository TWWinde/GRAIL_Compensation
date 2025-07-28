import torch
import torch.nn as nn
from collections import defaultdict
from models.resnet18 import get_axis_to_perm_ResNet18, get_module_by_name_ResNet18, axes2perm_to_perm2axes

class BaseResNet18Compression:
    """
    Base class for ResNet18 compression (folding or pruning).
    Unified apply() pipeline:
      - Iterate groups via axis_to_perm
      - Collect weights & biases
      - Call self.compress_or_prune() to get compressed params
      - Rebuild and replace modules
    """

    def __init__(self, model, min_channels=1, compression_ratio=0.5):
        self.model = model
        self.min_channels = min_channels
        self.keep_ratio = 1.0 - compression_ratio
        self.device = next(model.parameters()).device
        self._rename_layers(self.model)

    # --- Setup ---
    def _rename_layers(self, model):
        """Attach flat module dict for fast lookup"""
        model._module_dict = {name: module for name, module in model.named_modules()}

    # --- Abstract method ---
    def compress_or_prune(self, axes, params, **kwargs):
        """
        Must return: compressed_params (dict), optional_metadata
        Implemented differently by Folding and Pruning subclasses.
        """
        raise NotImplementedError

    # --- BN handling ---
    def _fold_bn_params(self, original_bn, cluster_labels, n_clusters):
        """Fuse BatchNorm params by averaging clusters"""
        device = original_bn.weight.device
        new_bn = nn.BatchNorm2d(n_clusters).to(device)

        for pname in ['weight', 'bias']:
            original = getattr(original_bn, pname).data
            fused = torch.stack([
                original[cluster_labels == k].mean() if (cluster_labels == k).any() else torch.tensor(0., device=device)
                for k in range(n_clusters)
            ])
            setattr(new_bn, pname, nn.Parameter(fused))

        for pname in ['running_mean', 'running_var']:
            original = getattr(original_bn, pname).data
            fused = torch.stack([
                original[cluster_labels == k].mean() if (cluster_labels == k).any() else torch.tensor(0., device=device)
                for k in range(n_clusters)
            ])
            getattr(new_bn, pname).data.copy_(fused)

        return new_bn

    def _copy_bn(self, old_bn, param_dict):
        new_bn = nn.BatchNorm2d(param_dict['weight'].shape[0]).to(self.device)
        for pname in ['weight', 'bias', 'running_mean', 'running_var']:
            if pname in param_dict:
                getattr(new_bn, pname).data.copy_(param_dict[pname])
        return new_bn

    # --- Module rebuild ---
    def _rebuild_module(self, old_module, param_dict, cluster_labels=None, n_clusters=None):
        # Conv2d
        if isinstance(old_module, nn.Conv2d):
            w = param_dict['weight']
            new_conv = nn.Conv2d(
                in_channels=w.shape[1], out_channels=w.shape[0],
                kernel_size=old_module.kernel_size, stride=old_module.stride,
                padding=old_module.padding, dilation=old_module.dilation,
                groups=old_module.groups, bias='bias' in param_dict,
                padding_mode=old_module.padding_mode
            ).to(self.device)
            new_conv.weight.data.copy_(w)
            if 'bias' in param_dict:
                new_conv.bias.data.copy_(param_dict['bias'])
            return new_conv

        # BatchNorm2d
        elif isinstance(old_module, nn.BatchNorm2d):
            return self._fold_bn_params(old_module, cluster_labels, n_clusters) \
                if cluster_labels is not None else self._copy_bn(old_module, param_dict)

        # Linear
        elif isinstance(old_module, nn.Linear):
            w = param_dict['weight']
            new_fc = nn.Linear(w.shape[1], w.shape[0]).to(self.device)
            new_fc.weight.data.copy_(w)
            if 'bias' in param_dict:
                new_fc.bias.data.copy_(param_dict['bias'])
            return new_fc

        return old_module

    def _replace(self, name, new_layer):
        """Replace module by name in nested model"""
        parts = name.split('.')
        mod = self.model
        for part in parts[:-1]:
            mod = getattr(mod, part) if not part.isdigit() else mod[int(part)]
        last = parts[-1]
        if last.isdigit():
            mod[int(last)] = new_layer
        else:
            setattr(mod, last, new_layer)

    # --- Unified apply() ---
    def apply(self):
        print(f"[INFO] Starting {self.__class__.__name__}...")
        axis_to_perm = get_axis_to_perm_ResNet18(override=False)
        perm_to_axes = axes2perm_to_perm2axes(axis_to_perm)

        for perm_id, axes in perm_to_axes.items():
            # --- Collect raw params ---
            raw_params = {}
            for module_name, axis in axes:
                module = get_module_by_name_ResNet18(self.model, module_name)
                weight = module.weight.data if hasattr(module, 'weight') else module.data
                raw_params[module_name] = weight

            # --- Call specific compression/pruning logic ---
            compressed_params, meta = self.compress_or_prune(axes, raw_params)

            # --- Rebuild modules ---
            param_groups = defaultdict(dict)
            for full_name, tensor in compressed_params.items():
                module_name, pname = full_name.rsplit('.', 1)
                param_groups[module_name][pname] = tensor

            for module_name, param_dict in param_groups.items():
                module = get_module_by_name_ResNet18(self.model, module_name)
                cluster_labels = meta.get('cluster_labels') if meta else None
                n_clusters = param_dict['weight'].shape[0]
                new_module = self._rebuild_module(module, param_dict, cluster_labels, n_clusters)
                self._replace(module_name, new_module)

        return self.model


