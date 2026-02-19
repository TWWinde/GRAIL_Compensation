"""
Compensation applied after pruning to adjust weights
"""
from __future__ import annotations
import torch
import torch.nn as nn
from collections import defaultdict

from typing import Dict, List

from compression.base_clip_vit import BaseCLIPViTCompression
from compression.base_resnet import BaseResNetCompression
from compression.base_preact_resnet import BasePreActResNetCompression
from compression.base_vit import BaseViTCompression

class ResNet18_PruneCompensation:
    """
    Compensation helper for ResNet-18 after Wanda pruning.
    Applies regression-based compensation using Gram matrices.
    """

    def __init__(self, model, ridge_lambda=1e-3):
        self.model = model
        self.ridge_lambda = ridge_lambda
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        
        self._named_modules = dict(model.named_modules())
        self._prune_entries = []
        self._grams = {}

    def _refresh(self):
        self._named_modules = dict(self.model.named_modules())

    def load_compression_state(self, entries):
        """load pruning metadata needed for compensation"""
        self._prune_entries = []
        for entry in entries:
            stored = {
                'layer_name': entry['layer_name'],
                'weight_full': entry['weight_full'].detach().cpu().clone(),
                'bias_full': entry['bias_full'].detach().cpu().clone() if entry['bias_full'] is not None else None,
                'original_shape': entry['original_shape'],
                'in_keep': entry['in_keep'],
                'out_keep': entry['out_keep'],
            }
            self._prune_entries.append(stored)
        self._refresh()

    def load_gram_stats(self, grams):
        """load Gram matrices measured before pruning"""
        self._grams = {name: G.detach().cpu().clone() for name, G in grams.items()}

    def _B_from_gram(self, layer_name: str, keep: torch.Tensor, device, dtype):
        """get compensation matrix B from Gram"""
        Gc = self._grams.get(layer_name)
        if Gc is None:
            return None

        if keep is None:
            keep_idx = torch.arange(Gc.size(0), device=device, dtype=torch.long)
        elif torch.is_tensor(keep):
            keep_idx = keep.to(device=device, dtype=torch.long)
        else:
            keep_idx = torch.as_tensor(keep, device=device, dtype=torch.long)

        if keep_idx.numel() == 0:
            return None

        G = Gc.to(device=device, dtype=dtype, non_blocking=True)
        G_PP = G.index_select(0, keep_idx).index_select(1, keep_idx)   # [K,K]
        G_PH = G.index_select(0, keep_idx)                             # [K,H]

        lam = self.ridge_lambda * (torch.diag(G_PP).mean().item() + 1e-12)
        A = G_PP + lam * torch.eye(keep_idx.numel(), device=device, dtype=dtype)

        try:
            X = torch.linalg.solve(A, G_PH)                            # [K,H]
        except RuntimeError:
            X = torch.linalg.lstsq(A, G_PH).solution                   # fallback

        return X.t().contiguous()                                   # [H,K]

    def _compensate_layer(self, entry):
        """compensate single layer"""
        layer_name = entry['layer_name']
        
        if layer_name not in self._named_modules:
            return
            
        layer = self._named_modules[layer_name]
        
        if  isinstance(layer, nn.Conv2d):
            self._compensate_conv_layer(entry, layer)
        elif isinstance(layer, nn.Linear):
            self._compensate_linear_layer(entry, layer)

    def _compensate_conv_layer(self, entry, layer):
        """compensate conv layer"""
        W_full = entry['weight_full'].to(self.device, self.dtype)

        out_keep = entry.get('out_keep')
        if out_keep is None:
            out_keep = torch.arange(W_full.shape[0], device=self.device, dtype=torch.long)
        elif torch.is_tensor(out_keep):
            out_keep = out_keep.to(self.device, dtype=torch.long)
        else:
            out_keep = torch.as_tensor(out_keep, device=self.device, dtype=torch.long)

        in_keep = entry.get('in_keep')
        if in_keep is None:
            in_keep = torch.arange(W_full.shape[1], device=self.device, dtype=torch.long)
        elif torch.is_tensor(in_keep):
            in_keep = in_keep.to(self.device, dtype=torch.long)
        else:
            in_keep = torch.as_tensor(in_keep, device=self.device, dtype=torch.long)

        W_sel = W_full.index_select(0, out_keep)                       # [O_keep, H_full, kH, kW]
        B = self._B_from_gram(entry['layer_name'], in_keep, self.device, self.dtype)

        if B is not None:
            W_new = torch.einsum('ohxy,hk->okxy', W_sel, B)       
        else:
            W_new = W_sel.index_select(1, in_keep)                     

        layer.weight.data.copy_(W_new.contiguous())

        if layer.bias is not None and entry['bias_full'] is not None:
            b_full = entry['bias_full'].to(self.device, self.dtype)
            layer.bias.data.copy_(b_full.index_select(0, out_keep))

    def _compensate_linear_layer(self, entry, layer):
        """compensate linear layer"""
        W_full = entry['weight_full'].to(self.device, self.dtype)

        out_keep = entry.get('out_keep')
        if out_keep is None:
            out_keep = torch.arange(W_full.shape[0], device=self.device, dtype=torch.long)
        elif torch.is_tensor(out_keep):
            out_keep = out_keep.to(self.device, dtype=torch.long)
        else:
            out_keep = torch.as_tensor(out_keep, device=self.device, dtype=torch.long)

        in_keep = entry.get('in_keep')
        if in_keep is None:
            in_keep = torch.arange(W_full.shape[1], device=self.device, dtype=torch.long)
        elif torch.is_tensor(in_keep):
            in_keep = in_keep.to(self.device, dtype=torch.long)
        else:
            in_keep = torch.as_tensor(in_keep, device=self.device, dtype=torch.long)

        W_sel = W_full.index_select(0, out_keep)                       # [O_keep, H_full]
        B = self._B_from_gram(entry['layer_name'], in_keep, self.device, self.dtype)

        if B is not None:
            W_new = torch.matmul(W_sel, B)                             
        else:
            W_new = W_sel.index_select(1, in_keep)                    

        layer.weight.data.copy_(W_new.contiguous())

        if layer.bias is not None and entry['bias_full'] is not None:
            b_full = entry['bias_full'].to(self.device, self.dtype)
            layer.bias.data.copy_(b_full.index_select(0, out_keep))

    @torch.no_grad()
    def apply(self):
        """apply compensation to all registered layers"""
        if not self._prune_entries:
            raise RuntimeError("No pruning metadata registered. Call `load_prune_state` first.")
            
        self._refresh()
        for entry in self._prune_entries:
            self._compensate_layer(entry) 
        self._refresh()
        return self.model




class PreActResNet18_PruneCompensation(ResNet18_PruneCompensation):
    """
    Compensation helper for PreActResNet-18 after Wanda pruning.
    Inherits the regression-based repair logic from the ResNet variant; the paired pruner
    supplies the required keep indices and Gram matrices for each layer.
    """

    pass


class Vit_PruneCompensation:
    """
    Compensation helper that is applied after ViT MLP blocks have already been pruned.

    The class expects pruning metadata (kept indices and original weights) collected while
    running `ViT_WandaPruning`, along with Gram matrices measured *before* the channels were
    removed. It then solves the same ridge-regression system used in Wanda-aware folding to
    rebuild the projection matrices in the pruned model without re-running the pruning step.
    """

    def __init__(self, model, ridge_lambda=1e-3):
        """
        Args:
            model: the pruned ViT model whose MLP blocks should be compensated in-place.
            ridge_lambda: regularization coefficient used when inverting Gram sub-matrices.
        """
        self.model = model
        self.ridge_lambda = ridge_lambda
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype

        self._named_modules = dict(model.named_modules())
        self._prune_entries = []
        self._grams = {}

    def _refresh(self):
        """Refresh internal cache after weights are rewritten."""
        self._named_modules = dict(self.model.named_modules())

    def load_compression_state(self, entries):
        """
        Register pruning metadata needed for compensation.

        Args:
            entries: iterable of dicts. Each dict must contain:
                - 'fc_name': str, module path of the pruned c_fc layer in the model.
                - 'proj_name': str, module path of the pruned c_proj layer in the model.
                - 'keep': 1D list/tensor of indices kept from the original hidden size.
                - 'weight_fc_full': torch.Tensor with the *pre-pruning* c_fc weight [H, C].
                - 'weight_proj_full': torch.Tensor with the *pre-pruning* c_proj weight [O, H].
              Optional keys:
                - 'bias_fc_full': torch.Tensor with the original c_fc bias [H].
                - 'bias_proj_full': torch.Tensor with the original c_proj bias [O].

        The method converts keep indices to `torch.LongTensor` and stores a CPU copy of the
        provided weights so they can be moved to whatever device compensation runs on.
        """
        self._prune_entries = []
        for entry in entries:
            required = {'fc_name', 'proj_name', 'keep', 'weight_fc_full', 'weight_proj_full'}
            missing = required - set(entry.keys())
            if missing:
                raise ValueError(f"Missing keys in prune entry: {missing}")

            keep = entry['keep']
            if not torch.is_tensor(keep):
                keep = torch.tensor(keep, dtype=torch.long)
            else:
                keep = keep.to(dtype=torch.long, device='cpu')

            stored = {
                'fc_name': entry['fc_name'],
                'proj_name': entry['proj_name'],
                'keep': keep.clone(),
                'weight_fc_full': entry['weight_fc_full'].detach().cpu().clone(),
                'weight_proj_full': entry['weight_proj_full'].detach().cpu().clone(),
            }
            if entry.get('bias_fc_full') is not None:
                stored['bias_fc_full'] = entry['bias_fc_full'].detach().cpu().clone()
            if entry.get('bias_proj_full') is not None:
                stored['bias_proj_full'] = entry['bias_proj_full'].detach().cpu().clone()

            self._prune_entries.append(stored)

        self._refresh()

    def load_gram_stats(self, grams):
        """
        Load Gram matrices measured before pruning.

        Args:
            grams: dict that maps `proj_name` (same key as provided in `load_prune_state`)
                   to a tensor of shape [H, H] containing the pre-pruning H^T H statistics.
                   Tensors are kept on CPU and moved to the target device during `apply()`.
        """
        self._grams = {name: G.detach().cpu().clone() for name, G in grams.items()}

    def _compensate_block(self, entry):
        """Apply regression-based compensation for a single (c_fc, c_proj) pair."""
        fc_name = entry['fc_name']
        proj_name = entry['proj_name']

        if fc_name not in self._named_modules or proj_name not in self._named_modules:
            raise KeyError(f"Expected modules '{fc_name}' and '{proj_name}' in the model.")

        keep = entry['keep'].to(device=self.device, dtype=torch.long)
        W_fc_full = entry['weight_fc_full'].to(device=self.device, dtype=self.dtype)
        W_proj_full = entry['weight_proj_full'].to(device=self.device, dtype=self.dtype)

        fc_layer = self._named_modules[fc_name]
        proj_layer = self._named_modules[proj_name]
        if not isinstance(fc_layer, nn.Linear) or not isinstance(proj_layer, nn.Linear):
            raise TypeError("Compensation expects both fc and proj modules to be nn.Linear.")

        # Ensure fc matches the kept channels (pure gather).
        new_fc_weight = W_fc_full.index_select(0, keep).contiguous()
        fc_layer.weight.data.copy_(new_fc_weight)
        if fc_layer.bias is not None and 'bias_fc_full' in entry:
            new_fc_bias = entry['bias_fc_full'].to(device=self.device, dtype=self.dtype)
            fc_layer.bias.data.copy_(new_fc_bias.index_select(0, keep).contiguous())

        G_cpu = self._grams.get(proj_name)
        if G_cpu is None:
            # Fallback: just gather the surviving columns.
            new_proj = W_proj_full.index_select(1, keep).contiguous()
        else:
            G = G_cpu.to(device=self.device, dtype=self.dtype, non_blocking=True)
            G_PP = G.index_select(0, keep).index_select(1, keep)
            G_PH = G.index_select(0, keep)
            lam = self.ridge_lambda * (torch.diag(G_PP).mean().item() + 1e-12)
            A = G_PP + lam * torch.eye(len(keep), device=self.device, dtype=self.dtype)
            try:
                X = torch.linalg.solve(A, G_PH)
            except RuntimeError:
                X = torch.linalg.lstsq(A, G_PH).solution
            B = X.t().contiguous()
            new_proj = (W_proj_full @ B).contiguous()

        proj_layer.weight.data.copy_(new_proj)
        if proj_layer.bias is not None:
            if 'bias_proj_full' in entry:
                proj_layer.bias.data.copy_(entry['bias_proj_full'].to(self.device, self.dtype))
            # else keep the existing bias (already pruned version)

    @torch.no_grad()
    def apply(self):
        """
        Run compensation for all registered MLP blocks.

        Returns:
            The same model instance with compensated projection weights.
        """
        if not self._prune_entries:
            raise RuntimeError("No pruning metadata registered. Call `load_prune_state` first.")

        self._refresh()
        for entry in self._prune_entries:
            self._compensate_block(entry)
        self._refresh()
        return self.model


class CLIPViT_PruneCompensation:
    """
    Compensation helper for CLIP ViT MLPs (c_fc + c_proj) after Wanda pruning.

    Uses pruning metadata captured by `CLIPViT_WandaPruning` along with pre-pruning Gram
    matrices to reconstruct projection weights via the Wanda regression step.
    """

    def __init__(self, model, ridge_lambda=1e-3):
        self.model = model
        self.ridge_lambda = ridge_lambda
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype

        self._named_modules = dict(model.named_modules())
        self._prune_entries = []
        self._grams = {}

    def _refresh(self):
        self._named_modules = dict(self.model.named_modules())

    def load_compression_state(self, entries):
        """
        Register pruning metadata required for compensation.

        Each entry must include:
            - fc_name / proj_name: module paths in the (already pruned) model.
            - keep: indices (original hidden basis) that were kept.
            - weight_fc_full: pre-pruning c_fc weight [H, C].
            - weight_proj_full: pre-pruning c_proj weight [O, H].
        Optional keys:
            - bias_fc_full: original c_fc bias [H].
            - bias_proj_full: original c_proj bias [O].
            - scores: Wanda scores [H] for Gram fallback weighting.
        """
        self._prune_entries = []
        for entry in entries:
            required = {'fc_name', 'proj_name', 'keep', 'weight_fc_full', 'weight_proj_full'}
            missing = required - set(entry.keys())
            if missing:
                raise ValueError(f"Missing keys in prune entry: {missing}")

            keep = entry['keep']
            if not torch.is_tensor(keep):
                keep = torch.tensor(keep, dtype=torch.long)
            else:
                keep = keep.to(dtype=torch.long, device='cpu')

            stored = {
                'fc_name': entry['fc_name'],
                'proj_name': entry['proj_name'],
                'keep': keep.clone(),
                'weight_fc_full': entry['weight_fc_full'].detach().cpu().clone(),
                'weight_proj_full': entry['weight_proj_full'].detach().cpu().clone(),
            }
            if entry.get('bias_fc_full') is not None:
                stored['bias_fc_full'] = entry['bias_fc_full'].detach().cpu().clone()
            if entry.get('bias_proj_full') is not None:
                stored['bias_proj_full'] = entry['bias_proj_full'].detach().cpu().clone()
            if entry.get('scores') is not None:
                stored['scores'] = entry['scores'].detach().cpu().clone()

            self._prune_entries.append(stored)

        self._refresh()

    def load_gram_stats(self, grams):
        """Load Gram matrices measured before pruning."""
        self._grams = {name: G.detach().cpu().clone() for name, G in grams.items()}

    def _compensate_block(self, entry):
        fc_name = entry['fc_name']
        proj_name = entry['proj_name']

        if fc_name not in self._named_modules or proj_name not in self._named_modules:
            raise KeyError(f"Expected modules '{fc_name}' and '{proj_name}'.")

        keep = entry['keep'].to(device=self.device, dtype=torch.long)
        W_fc_full = entry['weight_fc_full'].to(device=self.device, dtype=self.dtype)
        W_proj_full = entry['weight_proj_full'].to(device=self.device, dtype=self.dtype)

        fc_layer = self._named_modules[fc_name]
        proj_layer = self._named_modules[proj_name]
        if not isinstance(fc_layer, nn.Linear) or not isinstance(proj_layer, nn.Linear):
            raise TypeError("Compensation expects both fc and proj modules to be nn.Linear.")

        # Gather rows for c_fc directly.
        new_fc_weight = W_fc_full.index_select(0, keep).contiguous()
        fc_layer.weight.data.copy_(new_fc_weight)
        if fc_layer.bias is not None and 'bias_fc_full' in entry:
            bias_fc_full = entry['bias_fc_full'].to(device=self.device, dtype=self.dtype)
            fc_layer.bias.data.copy_(bias_fc_full.index_select(0, keep).contiguous())

        # Regression compensation for c_proj.
        G_cpu = self._grams.get(proj_name)
        if G_cpu is None or G_cpu.shape[0] != W_proj_full.shape[1]:
            new_proj = W_proj_full.index_select(1, keep).contiguous()
            scores = entry.get('scores')
            if scores is not None:
                scores = scores.to(device=self.device, dtype=self.dtype)
                scores_sel = scores.index_select(0, keep)
                if scores_sel.abs().sum() > 0:
                    alpha = (scores_sel / scores_sel.sum().clamp_min(1e-12)).view(1, -1)
                    averaged = (W_proj_full.index_select(1, keep) * alpha).sum(dim=1, keepdim=True)
                    new_proj = averaged.repeat(1, keep.numel()).contiguous()
        else:
            G = G_cpu.to(device=self.device, dtype=self.dtype, non_blocking=True)
            G_PP = G.index_select(0, keep).index_select(1, keep)
            G_PH = G.index_select(0, keep)
            lam = self.ridge_lambda * (torch.diag(G_PP).mean().item() + 1e-12)
            A = G_PP + lam * torch.eye(len(keep), device=self.device, dtype=self.dtype)
            try:
                X = torch.linalg.solve(A, G_PH)
            except RuntimeError:
                X = torch.linalg.lstsq(A, G_PH).solution
            B = X.t().contiguous()
            new_proj = (W_proj_full @ B).contiguous()

        proj_layer.weight.data.copy_(new_proj)
        if proj_layer.bias is not None and 'bias_proj_full' in entry:
            bias_proj_full = entry['bias_proj_full'].to(device=self.device, dtype=self.dtype)
            proj_layer.bias.data.copy_(bias_proj_full.contiguous())

    @torch.no_grad()
    def apply(self):
        if not self._prune_entries:
            raise RuntimeError("No pruning metadata registered. Call `load_prune_state` first.")

        self._refresh()
        for entry in self._prune_entries:
            self._compensate_block(entry)
        self._refresh()
        return self.model
