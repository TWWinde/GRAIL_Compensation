import torch
import torch.nn as nn

from models.resnet import get_module_by_name_ResNet, get_axis_to_perm_ResNet18
from models.preact_resnet import get_module_by_name_PreActResNet18, get_axis_to_perm_PreActResNet18

from compression.base_clip_vit import BaseCLIPViTCompression
from compression.base_resnet import BaseResNetCompression
from compression.base_preact_resnet import BasePreActResNetCompression
from compression.base_vit import BaseViTCompression

from utils.weight_clustering import axes2perm_to_perm2axes


class ResNet18_MagnitudePruning(BaseResNetCompression):
    def __init__(self, model, min_channels=1, compression_ratio=0.5, p=2):
        super().__init__(model, min_channels, compression_ratio)
        self.p = p
        self._hooks = []
        self.grams = {}        # module_name -> Gram matrix collected during calibration
        self.prune_entries = []  #

    # -------- Calibration (collect Gram) --------
    def _register_activation_hooks(self):
        self._grams_local = {}  

        def make_hook(name):
            def hook(mod, inputs):
                x = inputs[0]
                with torch.no_grad():
                    if isinstance(mod, nn.Conv2d):
                        
                        # collect Gram
                        N, C, H, W = x.shape
                        xt = x.float().permute(0, 2, 3, 1).reshape(-1, C)
                        G = (xt.t() @ xt).detach().cpu()
                        self._grams_local[name] = self._grams_local.get(name, 0) + G
                        
                    elif isinstance(mod, nn.Linear):
                        # collect Gramq
                        xt = x.float().reshape(-1, x.shape[-1])
                        G = (xt.t() @ xt).detach().cpu()
                        self._grams_local[name] = self._grams_local.get(name, 0) + G
                    else:
                        return
            return hook

        for lname, layer in self.model_layers.items():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                self._hooks.append(layer.register_forward_pre_hook(make_hook(lname)))

    def _clear_hooks(self):
        for h in self._hooks:
            try: h.remove()
            except: pass
        self._hooks = []

    @torch.no_grad()
    def run_calibration(self, dataloader, device, num_batches=10):
        self.model.eval().to(device)
        self._register_activation_hooks()
        seen = 0
        for batch in dataloader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            _ = self.model(x.to(device))
            seen += 1
            if seen >= num_batches:
                break
        self._clear_hooks()
        self.grams = {name: G.to(torch.float32) for name, G in self._grams_local.items()}
        self._grams_local = {}
        self.prune_entries = []
    
    def _record_keep(self, layer_name, in_keep=None, out_keep=None):
        """Optional: record which channels were kept per layer for analysis."""
        for prune_entry in self.prune_entries:
            if prune_entry['layer_name'] == layer_name:
                prune_entry['in_keep'] = in_keep.detach().cpu().clone() if in_keep is not None else None
                prune_entry['out_keep'] = out_keep.detach().cpu().clone() if out_keep is not None else None
           
                return

    def apply(self):
        self._save_prune_state()
        # Initial conv + BN
        conv1 = self.model_layers['conv1']
        k1 = max(int(conv1.out_channels * self.keep_ratio), self.min_channels)
        scores1 = torch.norm(conv1.weight.view(conv1.out_channels, -1), p=self.p, dim=1)
        keep = self._get_keep_indices(scores1, k1)
        self._rebuild_conv('conv1', out_keep=keep)
        self._adjust_bn('bn1', keep)
        self._record_keep('conv1', out_keep=keep)
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
                conv1_k = max(int(len(conv1_scores) * self.keep_ratio), self.min_channels)
                keep1 = self._get_keep_indices(conv1_scores, conv1_k)
                in_keep = prev_keep if i == 0 else keep2

                self._rebuild_conv(f"{prefix}.conv1", in_keep=in_keep, out_keep=keep1)
                self._adjust_bn(f"{prefix}.bn1", keep1)
                self._record_keep(f"{prefix}.conv1", in_keep=in_keep, out_keep=keep1)

                # conv2 pruning
                conv2 = block.conv2
                conv2_scores = torch.norm(conv2.weight.view(conv2.out_channels, -1), p=self.p, dim=1)
                conv2_k = max(int(len(conv2_scores) * self.keep_ratio), self.min_channels)

                if hasattr(block_module, 'downsample') and isinstance(block_module.downsample, nn.Sequential):
                    downsample_conv_name = f"{prefix}.downsample.0"
                    if downsample_conv_name in self.model_layers:
                        keep2 = self._get_keep_indices(conv2_scores, conv2_k)
                        self._rebuild_conv(downsample_conv_name, in_keep=prev_keep, out_keep=keep2)
                        self._adjust_bn(f"{prefix}.downsample.1", keep2)
                        self._record_keep(downsample_conv_name, in_keep=prev_keep, out_keep=keep2)
                    else:
                        keep2 = in_keep
                else:
                    keep2 = in_keep

                self._rebuild_conv(f"{prefix}.conv2", in_keep=keep1, out_keep=keep2)
                self._adjust_bn(f"{prefix}.bn2", keep2)
                self._record_keep(f"{prefix}.conv2", in_keep=keep1, out_keep=keep2)

                prev_keep = keep2

        # Final FC
        self._prune_linear("fc", prev_keep)
        self._record_keep('fc', in_keep=prev_keep)

        return self.model
    
    def _save_prune_state(self):
        """save pruning state for compensation later"""
        self.prune_entries = []
        
        for layer_name, layer in self.model_layers.items():
            if isinstance(layer, nn.Conv2d):
                prune_entry = {
                    'layer_name': layer_name,
                    'weight_full': layer.weight.data.detach().clone(),
                    'bias_full': layer.bias.data.detach().clone() if layer.bias is not None else None,
                    'original_shape': layer.weight.data.shape
                }
                self.prune_entries.append(prune_entry)

            elif isinstance(layer, nn.Linear):
                prune_entry = {
                    'layer_name': layer_name,
                    'weight_full': layer.weight.data.detach().clone(),
                    'bias_full': layer.bias.data.detach().clone() if layer.bias is not None else None,
                    'original_shape': layer.weight.data.shape
                }

                self.prune_entries.append(prune_entry)

    def get_compression_state(self):
        """save prune_entries """
        return self.prune_entries

    def get_gram_stats(self):
        return self.grams




# --- Magnitude pruning for CLIP ViT ---
class CLIPViT_MagnitudePruning(BaseCLIPViTCompression):
    def __init__(self, model, min_channels=1, compression_ratio=0.5, p=2):
        super().__init__(model, min_channels, compression_ratio)
        self.p = p
        self._hooks = []
        self._named_modules = dict(self.model.named_modules())
        self.grams = {}        # module_name -> Gram matrix collected during calibration
        self.prune_entries = []

    # -------------------- Calibration --------------------
    def _register_activation_hooks(self):
        """Collect Gram matrices for each Linear layer."""
        self._grams_local = {}

        def make_hook(name):
            def hook(mod, inputs):
                x = inputs[0]
                with torch.no_grad():
                    x2 = x.float().reshape(-1, x.shape[-1])
                    G = (x2.t() @ x2).detach().cpu()
                    self._grams_local[name] = self._grams_local.get(name, 0) + G
            return hook

        for name, mod in self._named_modules.items():
            if isinstance(mod, nn.Linear):
                self._hooks.append(mod.register_forward_pre_hook(make_hook(name)))

    def _clear_hooks(self):
        for h in self._hooks:
            try:
                h.remove()
            except Exception:
                pass
        self._hooks = []

    @torch.no_grad()
    def run_calibration(self, dataloader, device, num_batches=100, forward_fn=None):
        """
        Run several batches to accumulate Gram matrices for Linear layers.
        """
        self.model.eval().to(device)
        self._register_activation_hooks()

        fwd = forward_fn or getattr(self.model, "encode_image", None) or (lambda x: self.model(x))
        seen = 0
        for batch in dataloader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            fwd(x.to(device))
            seen += 1
            if seen >= num_batches:
                break

        self._clear_hooks()
        self.grams = {name: G.to(dtype=torch.float32) for name, G in self._grams_local.items()}
        self._grams_local = {}
        self.prune_entries = []

    def compress_function(self, axes, params):
        """
        Magnitude-based pruning for CLIP ViT MLP (c_fc + c_proj):
        - Rank c_fc output channels by Lp norm
        - Keep top-k channels according to keep_ratio
        - Apply the same selection to c_proj input channels
        """
        compressed = {}
        merge_sizes = {}

        module_fc, _ = axes[0]   # c_fc
        module_proj, _ = axes[1] # c_proj

        W_fc = params[module_fc]       # [hidden_dim, in_dim]
        W_proj = params[module_proj]   # [out_dim, hidden_dim]

        hidden_dim = W_fc.shape[0]
        keep_units = max(int(hidden_dim * self.keep_ratio), self.min_channels)

        norms = torch.norm(W_fc, dim=1, p=self.p)  # [hidden_dim]
        topk_indices = torch.topk(norms, keep_units, largest=True).indices.sort()[0].to(W_fc.device)

        prune_entry = {
            'fc_name': module_fc,
            'proj_name': module_proj,
            'keep': topk_indices.clone().cpu(),
            'weight_fc_full': W_fc.detach().clone(),
            'weight_proj_full': W_proj.detach().clone(),
        }
        if module_fc + '.bias' in params and params[module_fc + '.bias'] is not None:
            prune_entry['bias_fc_full'] = params[module_fc + '.bias'].detach().clone()
        if module_proj + '.bias' in params and params[module_proj + '.bias'] is not None:
            prune_entry['bias_proj_full'] = params[module_proj + '.bias'].detach().clone()
        self.prune_entries.append(prune_entry)

        new_fc = W_fc[topk_indices, :]
        new_proj = W_proj[:, topk_indices]

        compressed[module_fc + '.weight'] = new_fc
        compressed[module_proj + '.weight'] = new_proj

        if module_fc + '.bias' in params and params[module_fc + '.bias'] is not None:
            compressed[module_fc + '.bias'] = params[module_fc + '.bias'][topk_indices]
        if module_proj + '.bias' in params and params[module_proj + '.bias'] is not None:
            compressed[module_proj + '.bias'] = params[module_proj + '.bias']

        merge_sizes[module_fc] = new_fc.shape[0]
        merge_sizes[module_proj] = new_proj.shape[1]

        return compressed, merge_sizes

    def get_compression_state(self):
        """Return pruning metadata for downstream compensation."""
        return self.prune_entries

    def get_gram_stats(self):
        """Return Gram matrices collected during calibration."""
        return self.grams




class PreActResNet18_MagnitudePruning(BasePreActResNetCompression):
    def __init__(self, model, min_channels=1, compression_ratio=0.5, p=2):
        super().__init__(model, min_channels, compression_ratio)
        self.p = p

    def apply(self):
        # --- Initial conv (no BN at root in PreActResNet) ---
        conv1 = self.model_layers['conv1']
        k1 = max(int(conv1.out_channels * self.keep_ratio), self.min_channels)
        scores1 = torch.norm(conv1.weight.view(conv1.out_channels, -1), p=self.p, dim=1)
        keep = self._get_keep_indices(scores1, k1)
        self._rebuild_conv('conv1', out_keep=keep)
        prev_keep = keep

        # Residual blocks
        for layer in ['layer1', 'layer2', 'layer3', 'layer4']:
            blocks = self.model_layers[layer]

            for i, block in enumerate(blocks):
                prefix = f"{layer}.{i}"

                conv1_scores = torch.norm(block.conv1.weight.view(block.conv1.out_channels, -1), p=self.p, dim=1)
                conv1_k = max(int(len(conv1_scores) * self.keep_ratio), self.min_channels)
                keep1 = self._get_keep_indices(conv1_scores, conv1_k)
                in_keep = prev_keep

                self._rebuild_conv(f"{prefix}.conv1", in_keep=in_keep, out_keep=keep1)
                self._adjust_bn(f"{prefix}.bn1", in_keep)

                conv2_scores = torch.norm(block.conv2.weight.view(block.conv2.out_channels, -1), p=self.p, dim=1)
                conv2_k = max(int(len(conv2_scores) * self.keep_ratio), self.min_channels)

                if hasattr(block, 'shortcut') and isinstance(block.shortcut, nn.Sequential):
                    keep2 = self._get_keep_indices(conv2_scores, conv2_k)
                else:
                    keep2 = in_keep

                self._rebuild_conv(f"{prefix}.conv2", in_keep=keep1, out_keep=keep2)
                self._adjust_bn(f"{prefix}.bn2", keep1)

                if hasattr(block, 'shortcut') and isinstance(block.shortcut, nn.Sequential):
                    self._rebuild_conv(f"{prefix}.shortcut.0", in_keep=in_keep, out_keep=keep2)

                prev_keep = keep2

        self._adjust_bn("bn", prev_keep)
        self._prune_linear("linear", prev_keep)
        return self.model




class ViT_MagnitudePruning(BaseViTCompression):
    """
    Magnitude-based pruning for SimpleViT MLP (FeedForward layers):
    - Rank c_fc output channels by L2 norm
    - Keep top channels according to keep_ratio
    - Apply same selection to c_proj input channels
    """

    def __init__(self, model, min_channels=1, compression_ratio=0.5, p=2):
        super().__init__(model, min_channels, compression_ratio)
        self.p = p
        self._hooks = []
        self._named_modules = dict(self.model.named_modules())
        self.grams = {}        # module_name -> Gram matrix collected during calibration
        self.prune_entries = []  # reset per-apply; filled with metadata for compensation

    # -------------------- Calibration --------------------
    def _register_activation_hooks(self):
        """
        Collect Gram matrix.
        """
        self._grams_local = {}

        def make_hook(name):
            def hook(mod, inputs):
                x = inputs[0]
                with torch.no_grad():
                    # flatten all but last dim -> [T, C]
                    x2 = x.float().reshape(-1, x.shape[-1])
                    G = (x2.t() @ x2).detach().cpu()  
                    self._grams_local[name] = self._grams_local.get(name, 0) + G
            return hook

        # Attach to every Linear; we’ll only use entries for c_fc modules in compress()
        for name, mod in self._named_modules.items():
            if isinstance(mod, nn.Linear):
                self._hooks.append(mod.register_forward_pre_hook(make_hook(name)))

    def _clear_hooks(self):
        for h in self._hooks:
            try: h.remove()
            except: pass
        self._hooks = []

    @torch.no_grad()
    def run_calibration(self, dataloader, device, num_batches=50):
        """
        Run a few batches to estimate Gram Matrix for each Linear.
        Use a *clean/no-aug* loader if possible.
        """
        self.model.eval().to(device)
        self._register_activation_hooks()
        seen = 0
        for batch in dataloader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            _ = self.model(x.to(device))
            seen += 1
            if seen >= num_batches:
                break
        self._clear_hooks()

        # L2 = sqrt(sum of squares); clamp for numerical stability
        self.grams = {name: G.to(dtype=torch.float32) for name, G in self._grams_local.items()}
        self._grams_local = {}
        self.prune_entries = []

    def compress_function(self, axes, params):
        """
        Compress weights for SimpleViT MLP (c_fc + c_proj) using magnitude pruning.
        """
        compressed = {}
        merge_sizes = {}

        # --- Unpack module names ---
        module_fc = axes[0]
        module_proj = axes[1]

        # --- Extract weights ---
        W_fc = params[module_fc + '.weight']   # [hidden_dim, in_dim]
        W_proj = params[module_proj + '.weight']  # [out_dim, hidden_dim]

        # --- Determine number of channels to keep ---
        n_channels = W_fc.shape[0]
        keep_units = max(int(n_channels * self.keep_ratio), self.min_channels)

        # --- Compute L2 norm per output channel (row-wise) ---
        norms = torch.norm(W_fc, dim=1, p=self.p)  # [hidden_dim]
        topk_indices = torch.topk(norms, keep_units, largest=True).indices.sort()[0].to(W_fc.device)

        # --- Save prune entry for compensation later ---
        prune_entry = {
            'fc_name': module_fc,
            'proj_name': module_proj,
            'keep': topk_indices.clone().cpu(),
            'weight_fc_full': W_fc.detach().clone(),
            'weight_proj_full': W_proj.detach().clone(),
        }

        if module_fc + '.bias' in params and params[module_fc + '.bias'] is not None:
            prune_entry['bias_fc_full'] = params[module_fc + '.bias'].detach().clone()
        if module_proj + '.bias' in params and params[module_proj + '.bias'] is not None:
            prune_entry['bias_proj_full'] = params[module_proj + '.bias'].detach().clone()
        
        self.prune_entries.append(prune_entry)

        # --- Apply selection ---
        new_fc = W_fc[topk_indices, :]         # Reduce rows
        new_proj = W_proj[:, topk_indices]     # Reduce columns

        compressed[module_fc + '.weight'] = new_fc
        compressed[module_proj + '.weight'] = new_proj

        # --- Handle biases ---
        if module_fc + '.bias' in params and params[module_fc + '.bias'] is not None:
            compressed[module_fc + '.bias'] = params[module_fc + '.bias'][topk_indices]
        if module_proj + '.bias' in params and params[module_proj + '.bias'] is not None:
            compressed[module_proj + '.bias'] = params[module_proj + '.bias']

        # --- Track new sizes ---
        merge_sizes[module_fc] = new_fc.shape[0]
        merge_sizes[module_proj] = new_proj.shape[1]

        return compressed, merge_sizes
    
    def get_compression_state(self):
        """get prune_entries """
        return self.prune_entries

    def get_gram_stats(self):
        """get gram matrices """
        return self.grams
