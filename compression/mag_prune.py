import torch
import torch.nn as nn
from collections import defaultdict

from compression.base_clip_vit import BaseCLIPViTCompression
from compression.base_resnet import BaseResNet18Compression
from models.resnet18 import get_axis_to_perm_ResNet18, get_module_by_name_ResNet18, axes2perm_to_perm2axes


class ResNet18_MagnitudePruning:
    def __init__(self, model, p=1, min_channels=1, compression_ratio=0.5):
        print(f"[DEBUG] MagnitudePruning with compression_ratio={compression_ratio}, p={p}, min_channels={min_channels}")
        self.model = model.to("cuda" if torch.cuda.is_available() else "cpu")
        self.device = next(self.model.parameters()).device
        self.p = p
        self.min_channels = min_channels
        self.compression_ratio = compression_ratio
        self.model_layers = dict(self.model.named_modules())

    def _get_keep_indices(self, scores, k):
        return torch.argsort(scores, descending=True)[:k]

    def apply(self):
        # Prune initial conv and bn (less aggressive)
        conv1 = self.model_layers['conv1']
        k1 = max(int(conv1.out_channels * (1 - self.compression_ratio)), self.min_channels)
        scores1 = torch.norm(conv1.weight.view(conv1.out_channels, -1), p=self.p, dim=1)
        keep = self._get_keep_indices(scores1, k1)
        self._rebuild_conv('conv1', out_keep=keep)
        self._adjust_bn('bn1', keep)
        prev_keep = keep

        # Residual blocks
        for layer in ['layer1', 'layer2', 'layer3', 'layer4']:
            blocks = self.model_layers[layer]

            for i, block in enumerate(blocks):
                prefix = f"{layer}.{i}"
                block_module = self.model_layers[prefix]

                # conv1 pruning
                conv1 = block.conv1
                conv1_scores = torch.norm(conv1.weight.view(conv1.out_channels, -1), p=self.p, dim=1)
                conv1_k = max(int(len(conv1_scores) * (1 - self.compression_ratio)), self.min_channels)
                keep1 = self._get_keep_indices(conv1_scores, conv1_k)
                in_keep = prev_keep if i == 0 else keep2

                self._rebuild_conv(f"{prefix}.conv1", in_keep=in_keep, out_keep=keep1)
                self._adjust_bn(f"{prefix}.bn1", keep1)

                # conv2 pruning
                conv2 = block.conv2
                conv2_scores = torch.norm(conv2.weight.view(conv2.out_channels, -1), p=self.p, dim=1)
                conv2_k = max(int(len(conv2_scores) * (1 - self.compression_ratio)), self.min_channels)

                if hasattr(block_module, 'downsample') and isinstance(block_module.downsample, nn.Sequential):
                    downsample_conv_name = f"{prefix}.downsample.0"
                    if downsample_conv_name in self.model_layers:
                        keep2 = self._get_keep_indices(conv2_scores, conv2_k)
                        self._rebuild_conv(downsample_conv_name, in_keep=prev_keep, out_keep=keep2)
                        self._adjust_bn(f"{prefix}.downsample.1", keep2)
                    else:
                        keep2 = in_keep
                else:
                    keep2 = in_keep

                self._rebuild_conv(f"{prefix}.conv2", in_keep=keep1, out_keep=keep2)
                self._adjust_bn(f"{prefix}.bn2", keep2)

                prev_keep = keep2

        # Prune the final FC layer
        self._prune_linear("fc", prev_keep)
        # Do not reset BN stats here to preserve running_mean/var for pre-REPAIR evaluation
        return self.model

    def _prune_linear(self, name, keep):
        layer = self.model_layers[name]
        assert isinstance(layer, nn.Linear), f"{name} is not Linear but {type(layer)}"

        expected_in = len(keep)
        new_linear = nn.Linear(in_features=expected_in, out_features=layer.out_features).to(self.device)

        if layer.weight.shape[1] == expected_in:
            new_weight = layer.weight.clone()
        else:
            new_weight = layer.weight[:, keep].clone()
        new_linear.weight = nn.Parameter(new_weight)

        if layer.bias is not None:
            new_linear.bias = nn.Parameter(layer.bias.detach().clone())

        self._replace(name, new_linear)
        self.model_layers[name] = new_linear

    def _rebuild_conv(self, name, in_keep=None, out_keep=None):
        if name not in self.model_layers:
            return

        layer = self.model_layers[name]
        assert isinstance(layer, nn.Conv2d), f"{name} is not Conv2d but {type(layer)}"

        weight = layer.weight.detach()
        orig_out, orig_in = weight.shape[:2]

        out_indices = out_keep if out_keep is not None else torch.arange(orig_out, device=weight.device)
        in_indices = in_keep if in_keep is not None else torch.arange(orig_in, device=weight.device)

        new_weight = weight.index_select(0, out_indices).index_select(1, in_indices).clone()

        new_conv = nn.Conv2d(
            in_channels=len(in_indices),
            out_channels=len(out_indices),
            kernel_size=layer.kernel_size,
            stride=layer.stride,
            padding=layer.padding,
            dilation=layer.dilation,
            groups=layer.groups,
            bias=layer.bias is not None,
            padding_mode=layer.padding_mode
        ).to(self.device)

        new_conv.weight = nn.Parameter(new_weight)
        if layer.bias is not None:
            new_conv.bias = nn.Parameter(layer.bias[out_indices].clone())

        self._replace(name, new_conv)
        self.model_layers[name] = new_conv

    def _adjust_bn(self, name, keep):
        bn = self.model_layers[name]
        assert isinstance(bn, nn.BatchNorm2d), f"{name} is not BatchNorm2d but {type(bn)}"
        new_bn = nn.BatchNorm2d(len(keep)).to(self.device)
        # Copy affine params (gamma/beta)
        new_bn.weight = nn.Parameter(bn.weight[keep].detach().clone())
        new_bn.bias = nn.Parameter(bn.bias[keep].detach().clone())
        # Copy running stats to preserve pre-pruning activations stability
        new_bn.running_mean = bn.running_mean[keep].detach().clone()
        new_bn.running_var = bn.running_var[keep].detach().clone()
        self._replace(name, new_bn)
        self.model_layers[name] = new_bn

    def _replace(self, name, new_layer):
        parts = name.split('.')
        mod = self.model
        for part in parts[:-1]:
            mod = getattr(mod, part) if not part.isdigit() else mod[int(part)]
        last = parts[-1]
        if last.isdigit():
            mod[int(last)] = new_layer
        else:
            setattr(mod, last, new_layer)

    def _reset_bn_stats(self, module):
        # This can be called explicitly before REPAIR if needed
        for m in module.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.running_mean.zero_()
                m.running_var.fill_(1)











# --- Magnitude pruning for CLIP ViT ---
class CLIPViT_MagnitudePruning(BaseCLIPViTCompression):
    def compress_function(self, axes, params):
        """
            Perform magnitude-based channel pruning for CLIP ViT MLP (c_fc + c_proj):
            - Rank c_fc output channels by magnitude (L2 norm)
            - Keep top-k channels according to compression_ratio
            - Apply same selection to c_proj input channels
            """
        compressed = {}
        merge_sizes = {}

        # --- Unpack modules ---
        module_fc, _ = axes[0]  # c_fc (output channels)
        module_proj, _ = axes[1]  # c_proj (input channels)

        # --- Extract weights ---
        W_fc = params[module_fc]  # [hidden_dim, in_dim]
        W_proj = params[module_proj]  # [out_dim, hidden_dim]

        # --- Compute per-channel L2 norm ---
        norms = torch.norm(W_fc.view(W_fc.shape[0], -1), dim=1, p=2)  # [hidden_dim]

        # --- Determine number of channels to keep ---
        n_channels = W_fc.shape[0]
        k = max(int(n_channels * self.keep_ratio), self.min_channels)  # keep ratio
        topk_indices = torch.topk(norms, k=k, largest=True).indices

        # --- Sort indices for consistency ---
        topk_indices, _ = torch.sort(topk_indices)

        # --- Select pruned weights ---
        new_fc = W_fc[topk_indices, :]
        new_proj = W_proj[:, topk_indices]

        compressed[module_fc + '.weight'] = new_fc
        compressed[module_proj + '.weight'] = new_proj

        # --- Bias pruning
        if module_fc + '.bias' in params and params[module_fc + '.bias'] is not None:
            compressed[module_fc + '.bias'] = params[module_fc + '.bias'][topk_indices]
        if module_proj + '.bias' in params and params[module_proj + '.bias'] is not None:
            compressed[module_proj + '.bias'] = params[module_proj + '.bias']

        # --- Track new sizes
        merge_sizes[module_fc] = new_fc.shape[0]
        merge_sizes[module_proj] = new_proj.shape[1]

        return compressed, merge_sizes
