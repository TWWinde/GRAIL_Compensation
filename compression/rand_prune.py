import torch
import torch.nn as nn
import random

class ResNet18_RandomPruning:
    def __init__(self, model, min_channels=1, compression_ratio=0.5):
        print(f"[DEBUG] RandomPruning with compression_ratio={compression_ratio}, min_channels={min_channels}")
        self.model = model.to("cuda" if torch.cuda.is_available() else "cpu")
        self.device = next(self.model.parameters()).device
        self.min_channels = min_channels
        self.compression_ratio = compression_ratio
        self.model_layers = dict(self.model.named_modules())

    def _get_random_indices(self, size, k):
        indices = list(range(size))
        random.shuffle(indices)
        return torch.tensor(indices[:k], device=self.device)

    def apply(self):
        # Prune initial conv and bn (less aggressive)
        conv1 = self.model_layers['conv1']
        k1 = max(int(conv1.out_channels * (1 - self.compression_ratio)), self.min_channels)
        keep = self._get_random_indices(conv1.out_channels, k1)
        self._rebuild_conv('conv1', out_keep=keep)
        self._adjust_bn('bn1', keep)
        prev_keep = keep

        for layer in ['layer1', 'layer2', 'layer3', 'layer4']:
            blocks = self.model_layers[layer]
            for i, block in enumerate(blocks):
                prefix = f"{layer}.{i}"
                block_module = self.model_layers[prefix]

                # conv1 pruning
                conv1 = block.conv1
                conv1_k = max(int(conv1.out_channels * (1 - self.compression_ratio)), self.min_channels)
                keep1 = self._get_random_indices(conv1.out_channels, conv1_k)
                in_keep = prev_keep if i == 0 else keep2

                self._rebuild_conv(f"{prefix}.conv1", in_keep=in_keep, out_keep=keep1)
                self._adjust_bn(f"{prefix}.bn1", keep1)

                # conv2 pruning
                conv2 = block.conv2
                conv2_k = max(int(conv2.out_channels * (1 - self.compression_ratio)), self.min_channels)

                if hasattr(block_module, 'downsample') and isinstance(block_module.downsample, nn.Sequential) and len(
                        block_module.downsample) > 0:
                    keep2 = self._get_random_indices(conv2.out_channels, conv2_k)
                else:
                    keep2 = in_keep

                self._rebuild_conv(f"{prefix}.conv2", in_keep=keep1, out_keep=keep2)
                self._adjust_bn(f"{prefix}.bn2", keep2)

                # Downsample
                if hasattr(block_module, 'downsample') and isinstance(block_module.downsample, nn.Sequential) and len(
                        block_module.downsample) > 0:
                    self._rebuild_conv(f"{prefix}.downsample.0", in_keep=prev_keep, out_keep=keep2)
                    self._adjust_bn(f"{prefix}.downsample.1", keep2)

                prev_keep = keep2

        # Prune the final FC layer
        self._prune_linear("fc", prev_keep)
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
        new_bn.weight = nn.Parameter(bn.weight[keep].detach().clone())
        new_bn.bias = nn.Parameter(bn.bias[keep].detach().clone())
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
