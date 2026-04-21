import torch
import time

class MagnitudeCompensator:
    def __init__(self, model, ridge_lambda=1e-3):
        self.model = model
        self.device = next(model.parameters()).device
        self.ridge_lambda = ridge_lambda
        self.grams = {}
        self.prune_entries = []

    def load_gram_stats(self, grams):
        self.grams = grams

    def load_compression_state(self, entries):
        self.prune_entries = entries

    def compensate(self):
        print("   🔧 Running Compensation...")
        for entry in self.prune_entries:
            name = entry['name']
            if name not in self.grams:
                print(f"⚠️ No Gram matrix for {name}, skipping.")
                continue
                
            print(f"     Compensating {name}...")
            module = entry['module']
            W_full = entry['weight_full'].to(self.device)
            G = self.grams[name].to(self.device).float()
            
            if torch.isnan(G).any() or torch.isinf(G).any():
                print(f"❌ FATAL: Gram matrix for {name} contains NaN/Inf!")
                print(f"   Stats: min={G.min()}, max={G.max()}, mean={G.mean()}")
                continue
            
            # Determine if we are doing Unstructured (Masking) or Structured (Physical) compensation
            # If weight shapes differ, it must be Structured Pruning (Physical).
            is_physical_pruning = (module.weight.shape != W_full.shape)
            
            if entry.get('mask') is not None and not is_physical_pruning:
                # Unstructured / Row-wise Compensation
                mask = entry['mask'].to(self.device)
                W_new = torch.zeros_like(module.weight.data)
                
                rows = W_full.shape[0]
                t_start = time.time()
                for r in range(rows):
                    if r % 100 == 0:
                        print(f"       Row {r}/{rows}...", end='\r')
                    
                    # Find kept columns for this row (where mask is False, i.e., unpruned)
                    # Note: In prune_magnitude, W_mask is True where KEPT.
                    # But here 'mask' key in entry is stored as ~W_mask (True where PRUNED).
                    
                    if mask.dim() == 1:
                        # Shared mask for all rows (Column-wise masking)
                        # mask is True where pruned.
                        in_keep_row = torch.where(~mask)[0]
                    else:
                        # Per-element mask
                        in_keep_row = torch.where(~mask[r])[0]
                    
                    if len(in_keep_row) == 0:
                        continue

                    # DEBUG: Check shapes
                    if r == 0: # Print only for first row to avoid spam
                        print(f"       [DEBUG] {name}: G.shape={G.shape}, mask.shape={mask.shape}, in_keep_row max={in_keep_row.max() if len(in_keep_row)>0 else 'None'}")

                    # Extract sub-Gram matrix for this row's kept features
                    G_sub = G.index_select(0, in_keep_row).index_select(1, in_keep_row)
                    
                    # Target vector: G[S, :] @ w_orig.T
                    # We want to reconstruct Y = X @ w_orig.T
                    # We solve w_new @ X_S.T = Y  => X_S @ w_new.T = X @ w_orig.T
                    # Multiply by X_S.T: X_S.T @ X_S @ w_new.T = X_S.T @ X @ w_orig.T
                    # G_sub @ w_new.T = G[S, :] @ w_orig.T
                    
                    w_orig_row = W_full[r, :].float()
                    b = G.index_select(0, in_keep_row) @ w_orig_row
                    
                    # Ridge Regression
                    K = in_keep_row.numel()
                    # Force float32 for stability
                    G_sub = G_sub.float()
                    b = b.float()
                    
                    lam = self.ridge_lambda * (torch.diag(G_sub).mean() + 1e-3)
                    A = G_sub + lam * torch.eye(K, device=self.device).float()
                    
                    try:
                        w_new_row = torch.linalg.solve(A, b)
                    except RuntimeError:
                        w_new_row = torch.linalg.lstsq(A, b).solution
                        
                    W_new[r, in_keep_row] = w_new_row.to(W_new.dtype)
                
                torch.cuda.synchronize()
                t_end = time.time()
                print(f"       Compensated {rows} rows in {t_end - t_start:.2f}s")
                module.weight.data = W_new
                
            else:
                # Structured / Column-wise Compensation (Legacy)
                in_keep = entry['in_keep'].to(self.device)
                
                # Ridge Regression
                G_PP = G.index_select(0, in_keep).index_select(1, in_keep).float()
                G_PH = G.index_select(0, in_keep).float()
                
                K = in_keep.numel()
                lam = self.ridge_lambda * (torch.diag(G_PP).mean() + 1e-3)
                A = G_PP + lam * torch.eye(K, device=self.device).float().float()
                
                try:
                    M = torch.linalg.solve(A, G_PH)
                except RuntimeError:
                    M = torch.linalg.lstsq(A, G_PH).solution
                
                if torch.isnan(M).any() or torch.isinf(M).any():
                    print(f"       [DEBUG] {name}: M has NaNs after solve! Skipping compensation for this layer.")
                    continue

                W_new = W_full.float() @ M.t()
                
                # Assign compensated weights
                # Since we are doing Structured Pruning and running AFTER compress, 
                # W_new MUST match module.weight.shape.
                # If not, let PyTorch raise a shape mismatch error so we can debug.
                
                print(f"       [DEBUG] Assigning weights for {name}")
                print(f"         W_new dtype: {W_new.dtype}")
                print(f"         Module weight dtype: {module.weight.dtype}")
                
                module.weight.data = W_new.to(module.weight.dtype)
                print(f"         Assigned weight dtype: {module.weight.data.dtype}")
                
                # Bias Compensation Correction (REMOVED)
                # Logs showed correction is negligible (~0.0000) and user reported it hurts performance.
                # Weight Compensation alone preserves the mean sufficiently.
            
        print("   ✅ Compensation Done.")
