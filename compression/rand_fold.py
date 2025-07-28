import torch
import copy
from copy import deepcopy
import torch.nn as nn
from collections import defaultdict
from models.resnet18 import get_axis_to_perm_ResNet18, get_module_by_name_ResNet18
from utils.weight_clustering import axes2perm_to_perm2axes, compress_weight_clustering


class ResNet18_RandomFolding:
    def __init__(self, model, normalize=False, min_channels=1, compression_ratio=0.5):
        print(f"[DEBUG] RandomFolding with compression_ratio={compression_ratio}, normalize={normalize}, min_channels={min_channels}")
        self.model = model
        self.normalize = normalize
        self.min_channels = min_channels
        self.compression_ratio = 1.0 - compression_ratio
        self.device = next(model.parameters()).device

        self._rename_layers(self.model)

    def _rename_layers(self, model):
        for name, module in model.named_modules():
            if not hasattr(model, '_module_dict'):
                model._module_dict = {}
            model._module_dict[name] = module

    def _fold_bn_params(self, original_bn, cluster_labels, n_clusters):
        device = original_bn.weight.device
        new_bn = nn.BatchNorm2d(n_clusters).to(device)

        for param_name in ['weight', 'bias']:
            original = getattr(original_bn, param_name).data
            fused = torch.zeros(n_clusters, device=device)
            for k in range(n_clusters):
                mask = cluster_labels == k
                if mask.sum() > 0:
                    fused[k] = original[mask].mean()
            setattr(new_bn, param_name, nn.Parameter(fused))

        for stat_name in ['running_mean', 'running_var']:
            original = getattr(original_bn, stat_name).data
            fused = torch.zeros(n_clusters, device=device)
            for k in range(n_clusters):
                mask = cluster_labels == k
                if mask.sum() > 0:
                    fused[k] = original[mask].mean()
            getattr(new_bn, stat_name).data.copy_(fused)

        return new_bn

    def _rebuild_module(self, name, old_module, param_dict, cluster_labels=None, n_clusters=None):
        if isinstance(old_module, nn.Conv2d):
            new_weight = param_dict.get('weight')
            new_out_channels = new_weight.shape[0]
            new_in_channels = new_weight.shape[1]
            new_conv = nn.Conv2d(
                in_channels=new_in_channels,
                out_channels=new_out_channels,
                kernel_size=old_module.kernel_size,
                stride=old_module.stride,
                padding=old_module.padding,
                dilation=old_module.dilation,
                groups=old_module.groups,
                bias='bias' in param_dict,
                padding_mode=old_module.padding_mode
            ).to(self.device)
            new_conv.weight.data = param_dict['weight'].clone()
            if 'bias' in param_dict:
                new_conv.bias.data = param_dict['bias'].clone()
            return new_conv

        elif isinstance(old_module, nn.BatchNorm2d):
            if cluster_labels is not None and n_clusters is not None:
                return self._fold_bn_params(old_module, cluster_labels, n_clusters)
            else:
                new_num_features = param_dict['weight'].shape[0]
                new_bn = nn.BatchNorm2d(new_num_features).to(self.device)
                for pname in ['weight', 'bias', 'running_mean', 'running_var']:
                    if pname in param_dict:
                        getattr(new_bn, pname).data = param_dict[pname].clone()
                return new_bn

        elif isinstance(old_module, nn.Linear):
            new_weight = param_dict.get('weight')
            new_out_features = new_weight.shape[0]
            new_in_features = new_weight.shape[1]
            new_fc = nn.Linear(new_in_features, new_out_features).to(self.device)
            new_fc.weight.data = param_dict['weight'].clone()
            if 'bias' in param_dict:
                new_fc.bias.data = param_dict['bias'].clone()
            return new_fc

        return old_module

    def apply(self):
        print("Starting model folding with random clustering...")
        axis_to_perm = get_axis_to_perm_ResNet18(override=False)
        perm_to_axes = axes2perm_to_perm2axes(axis_to_perm)

        for perm_id, axes in perm_to_axes.items():
            features = []
            raw_params = {}
            module_offsets = {}
            offset = 0

            for module_name, axis in axes:
                module = get_module_by_name_ResNet18(self.model, module_name)
                weight = module.weight.data if hasattr(module, 'weight') else module.data
                raw_params[module_name] = weight
                weight = weight.transpose(0, axis).contiguous()
                n_channels = weight.shape[0]
                reshaped = weight.view(n_channels, -1)
                features.append(reshaped)
                module_offsets[module_name] = (offset, offset + n_channels)
                offset += n_channels

            all_features = torch.cat(features, dim=1)
            n_channels = all_features.shape[0]
            n_clusters = max(int(n_channels * self.compression_ratio), self.min_channels)

            # Random cluster assignment
            clustering_result = torch.randint(low=0, high=n_clusters, size=(n_channels,), dtype=torch.int)

            compressed_params, merge_sizes = compress_weight_clustering({perm_id: axes}, raw_params,
                                                                         max_ratio=self.compression_ratio, threshold=0.1,
                                                                         hooks=None, approx_repair=self.normalize,
                                                                         merge_layer=None, custom_merger=None)

            param_groups = defaultdict(dict)
            for full_name, tensor in compressed_params.items():
                module_name, param_name = full_name.rsplit('.', 1)
                param_groups[module_name][param_name] = tensor

            for module_name, param_dict in param_groups.items():
                module = get_module_by_name_ResNet18(self.model, module_name)
                cluster_labels = None
                if module_name in module_offsets:
                    start, end = module_offsets[module_name]
                    cluster_labels = clustering_result[start:end]

                new_module = self._rebuild_module(module_name, module, param_dict, cluster_labels, n_clusters)

                parent_name = '.'.join(module_name.split('.')[:-1])
                attr_name = module_name.split('.')[-1]
                if parent_name:
                    parent = get_module_by_name_ResNet18(self.model, parent_name)
                    setattr(parent, attr_name, new_module)
                else:
                    setattr(self.model, attr_name, new_module)

        return self.model
