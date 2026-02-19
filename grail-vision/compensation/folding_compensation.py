import torch
import torch.nn as nn


class ResNet18_FoldingCompensation:
    """
    Compensation helper applied after folding when we still have snapshots of the
    original consumer weights (before their input channels were merged).
    For each consumer layer we solve
        B* = G R^T (R G R^T + λ I)^{-1}
    and rebuild the folded weights via
        W_new = W_full @ B*
    (Conv2d handled via tensordot over the input-channel dimension).
    """

    def __init__(self, model, ridge_lambda=1e-3):
        self.model = model
        self.ridge_lambda = ridge_lambda
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype

        self._named_modules = dict(model.named_modules())
        self._fold_entries = []   # list of dicts with weight snapshots + R
        self._grams = {}

    def _refresh(self):
        self._named_modules = dict(self.model.named_modules())

    def load_compression_state(self, entries):
        """
        entries need fields:
            - 'layer_name': consumer module name
            - 'weight_full': Tensor of original weights [O,H,...]
            - 'bias_full': Tensor or None (original bias)
            - 'original_shape': tuple (unused but kept for debugging)
            - 'merge_matrix': Tensor R [K,H]
        """
        self._fold_entries = []
        for e in entries:
            stored = {
                'layer_name': e['layer_name'],
                'weight_full': e['weight_full'].detach().cpu().clone(),
                'bias_full': None if e.get('bias_full') is None else e['bias_full'].detach().cpu().clone(),
                'original_shape': tuple(e.get('original_shape', ())),
                'merge_matrix': e['merge_matrix'].detach().cpu().clone(),
                'output_merge_matrix': None if e.get('output_merge_matrix') is None else e['output_merge_matrix'].detach().cpu().clone()
            }
            self._fold_entries.append(stored)
        self._refresh()

    def load_gram_stats(self, grams):
        """Register Gram matrices measured BEFORE folding (dict name -> tensor)."""
        self._grams = {name: G.detach().cpu().clone() for name, G in grams.items()}

    # -------- core math: B* = G R^T (R G R^T + λ I)^(-1) --------
    def _B_from_gram(self, layer_name: str, R: torch.Tensor):
        G = self._grams.get(layer_name, None)
        if G is None:
            return None

        G = G.to(device=self.device, dtype=torch.float64, non_blocking=True)
        R = R.to(device=self.device, dtype=torch.float64, non_blocking=True)
        Rt = R.t()

        RGRT = R @ G @ Rt
        lam = self.ridge_lambda * (torch.diag(RGRT).mean().item() + 1e-12)
        M = RGRT + lam * torch.eye(RGRT.shape[0], dtype=torch.float64, device=self.device)

        X = G @ Rt  # [H,K]
        try:
            L = torch.linalg.cholesky(M)
            Z = torch.cholesky_solve(X.t(), L)  # [K,H]
        except RuntimeError:
            Z = torch.linalg.lstsq(M, X.t()).solution
        return Z.t().contiguous().to(self.dtype)  # [H,K]

    def _fallback_B(self, R: torch.Tensor):
        """Data-free fallback using simple averaging implied by merge matrix."""
        return R.t().contiguous().to(self.dtype)

    # -------- layer-wise reconstruction --------
    def _apply_output_merge(self, W, output_merge):
        """Map original output dimension to folded one if merge matrix is provided."""
        if output_merge is None:
            return W
        S = output_merge.to(self.device, self.dtype)
        if S.shape[1] != W.shape[0]:
            # Already folded or mismatch; skip applying this merge
            return W
        orig_shape = W.shape
        if W.dim() == 4:
            O, _, kh, kw = orig_shape
            merged = S @ W.view(O, -1)                       # [O_new, H*kh*kw]
            return merged.view(S.shape[0], orig_shape[1], kh, kw)
        else:
            O = orig_shape[0]
            merged = S @ W.view(O, -1)
            return merged.view(S.shape[0], *orig_shape[1:])

    def _rebuild_conv(self, entry, layer):
        W_full = entry['weight_full'].to(self.device, self.dtype)      # [O,H,kh,kw]
        R = entry['merge_matrix'].to(self.device, self.dtype)          # [K,H]

        B = self._B_from_gram(entry['layer_name'], R)
        if B is None:
            B = self._fallback_B(R)
        else:
            B = B.to(self.device, self.dtype)

        # [O,H,kh,kw] x [H,K] -> [O,K,kh,kw]
        W_new = torch.tensordot(W_full, B, dims=([1], [0]))
        if W_new.dim() == 4:
            W_new = W_new.permute(0, 3, 1, 2).contiguous()
        W_new = self._apply_output_merge(W_new, entry.get('output_merge_matrix'))
        layer.weight.data.copy_(W_new)

        if layer.bias is not None and entry['bias_full'] is not None:
            layer.bias.data.copy_(entry['bias_full'].to(self.device, self.dtype))

    def _rebuild_linear(self, entry, layer):
        W_full = entry['weight_full'].to(self.device, self.dtype)  # [O,H]
        R = entry['merge_matrix'].to(self.device, self.dtype)      # [K,H]

        B = self._B_from_gram(entry['layer_name'], R)
        if B is None:
            B = self._fallback_B(R)
        else:
            B = B.to(self.device, self.dtype)

        W_new = torch.matmul(W_full, B)  # [O,K]
        W_new = self._apply_output_merge(W_new, entry.get('output_merge_matrix'))
        layer.weight.data.copy_(W_new)

        if layer.bias is not None and entry['bias_full'] is not None:
            layer.bias.data.copy_(entry['bias_full'].to(self.device, self.dtype))

    def _compensate_layer(self, entry):
        lname = entry['layer_name']
        if lname not in self._named_modules:
            print(f"[WARN] consumer layer {lname} not found; skip.")
            return
        layer = self._named_modules[lname]
        if isinstance(layer, nn.Conv2d):
            self._rebuild_conv(entry, layer)
        elif isinstance(layer, nn.Linear):
            self._rebuild_linear(entry, layer)
        else:
            print(f"[WARN] unsupported consumer type for folding compensation: {type(layer)}")

    @torch.no_grad()
    def apply(self):
        if not self._fold_entries:
            raise RuntimeError("No folding metadata registered. Call `load_fold_state` first.")
        self._refresh()
        for entry in self._fold_entries:
            self._compensate_layer(entry)
        self._refresh()
        return self.model



class Vit_FoldingCompensation:
    """
    Compensation helper applied after folding SimpleViT MLP blocks.
    Expects merge metadata (R matrices) plus original projection weights recorded before folding.
    """

    def __init__(self, model, ridge_lambda=1e-3):
        self.model = model
        self.ridge_lambda = ridge_lambda
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype

        self._named_modules = dict(model.named_modules())
        self._fold_entries = []
        self._grams = {}

    def _refresh(self):
        self._named_modules = dict(self.model.named_modules())

    def load_compression_state(self, entries):
        """
        entries: iterable of dicts with fields
            - layer_name: consumer module (c_proj)
            - weight_full: original projection weight [O, H]
            - bias_full: optional original bias [O]
            - merge_matrix: folding matrix R [K, H]
        """
        self._fold_entries = []
        for e in entries:
            stored = {
                'layer_name': e['layer_name'],
                'weight_full': e['weight_full'].detach().cpu().clone(),
                'bias_full': None if e.get('bias_full') is None else e['bias_full'].detach().cpu().clone(),
                'merge_matrix': e['merge_matrix'].detach().cpu().clone()
            }
            self._fold_entries.append(stored)
        self._refresh()

    def load_gram_stats(self, grams):
        self._grams = {name: G.detach().cpu().clone() for name, G in grams.items()}

    def _B_from_gram(self, layer_name, R):
        Gc = self._grams.get(layer_name)
        if Gc is None:
            return None
        G = Gc.to(device=self.device, dtype=torch.float64, non_blocking=True)
        R = R.to(device=self.device, dtype=torch.float64, non_blocking=True)
        Rt = R.t()
        RGRT = R @ G @ Rt
        lam = self.ridge_lambda * (torch.diag(RGRT).mean().item() + 1e-12)
        M = RGRT + lam * torch.eye(RGRT.shape[0], dtype=torch.float64, device=self.device)
        X = G @ Rt
        try:
            L = torch.linalg.cholesky(M)
            Z = torch.cholesky_solve(X.t(), L)
        except RuntimeError:
            Z = torch.linalg.lstsq(M, X.t()).solution
        return Z.t().contiguous().to(self.dtype)

    def _fallback_B(self, R):
        return R.t().contiguous().to(self.dtype)

    def _rebuild_linear(self, entry, layer):
        W_full = entry['weight_full'].to(self.device, self.dtype)
        R = entry['merge_matrix'].to(self.device, self.dtype)

        B = self._B_from_gram(entry['layer_name'], R)
        if B is None:
            B = self._fallback_B(R)

        W_new = torch.matmul(W_full, B)
        layer.weight.data.copy_(W_new)

        if layer.bias is not None and entry['bias_full'] is not None:
            layer.bias.data.copy_(entry['bias_full'].to(self.device, self.dtype))

    def _compensate_layer(self, entry):
        lname = entry['layer_name']
        if lname not in self._named_modules:
            print(f"[WARN] ViT compensation skipped missing layer {lname}.")
            return
        layer = self._named_modules[lname]
        if not isinstance(layer, nn.Linear):
            print(f"[WARN] Expected Linear layer for {lname}, found {type(layer)}. Skipping.")
            return
        self._rebuild_linear(entry, layer)

    @torch.no_grad()
    def apply(self):
        if not self._fold_entries:
            raise RuntimeError("No folding metadata registered. Call `load_compression_state` first.")
        self._refresh()
        for entry in self._fold_entries:
            self._compensate_layer(entry)
        self._refresh()
        return self.model


class CLIPViT_FoldingCompensation:
    """
    Compensation helper for CLIP ViT folding using stored pre-fold projection weights.
    """

    def __init__(self, model, ridge_lambda=1e-3):
        self.model = model
        self.ridge_lambda = ridge_lambda
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype

        self._named_modules = dict(model.named_modules())
        self._fold_entries = []
        self._grams = {}

    def _refresh(self):
        self._named_modules = dict(self.model.named_modules())

    def load_compression_state(self, entries):
        """
        entries: list of dicts with keys:
            - layer_name: projection module name
            - weight_full: Tensor [O, H] before folding
            - bias_full: optional tensor [O]
            - merge_matrix: Tensor R [K, H]
        """
        self._fold_entries = []
        for e in entries:
            stored = {
                'layer_name': e['layer_name'],
                'weight_full': e['weight_full'].detach().cpu().clone(),
                'bias_full': None if e.get('bias_full') is None else e['bias_full'].detach().cpu().clone(),
                'merge_matrix': e['merge_matrix'].detach().cpu().clone()
            }
            self._fold_entries.append(stored)
        self._refresh()

    def load_gram_stats(self, grams):
        self._grams = {name: G.detach().cpu().clone() for name, G in grams.items()}

    def _B_from_gram(self, layer_name, R):
        Gc = self._grams.get(layer_name)
        if Gc is None:
            return None
        G = Gc.to(device=self.device, dtype=torch.float64, non_blocking=True)
        R = R.to(device=self.device, dtype=torch.float64, non_blocking=True)
        Rt = R.t()
        RGRT = R @ G @ Rt
        lam = self.ridge_lambda * (torch.diag(RGRT).mean().item() + 1e-12)
        M = RGRT + lam * torch.eye(RGRT.shape[0], dtype=torch.float64, device=self.device)
        X = G @ Rt
        try:
            L = torch.linalg.cholesky(M)
            Z = torch.cholesky_solve(X.t(), L)
        except RuntimeError:
            Z = torch.linalg.lstsq(M, X.t()).solution
        return Z.t().contiguous().to(self.dtype)

    def _fallback_B(self, R):
        return R.t().contiguous().to(self.dtype)

    def _compensate_layer(self, entry):
        lname = entry['layer_name']
        if lname not in self._named_modules:
            print(f"[WARN] CLIP ViT compensation skipped missing layer {lname}.")
            return
        layer = self._named_modules[lname]
        if not isinstance(layer, nn.Linear):
            print(f"[WARN] Expected Linear layer for {lname}, found {type(layer)}. Skipping.")
            return

        W_full = entry['weight_full'].to(self.device, self.dtype)
        R = entry['merge_matrix'].to(self.device, self.dtype)
        B = self._B_from_gram(lname, R)
        if B is None:
            B = self._fallback_B(R)

        W_new = torch.matmul(W_full, B)
        layer.weight.data.copy_(W_new)

        if layer.bias is not None and entry['bias_full'] is not None:
            layer.bias.data.copy_(entry['bias_full'].to(self.device, self.dtype))

    @torch.no_grad()
    def apply(self):
        if not self._fold_entries:
            raise RuntimeError("No folding metadata registered. Call `load_compression_state` first.")
        self._refresh()
        for entry in self._fold_entries:
            self._compensate_layer(entry)
        self._refresh()
        return self.model
