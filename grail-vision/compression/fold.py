import torch
import torch.nn as nn
from collections import defaultdict

from models.resnet import get_module_by_name_ResNet, get_axis_to_perm_ResNet18
from models.preact_resnet import get_module_by_name_PreActResNet18, get_axis_to_perm_PreActResNet18
from models.clip_vit import get_axis_to_perm_ViT, get_module_by_name_ViT
from utils.weight_clustering import WeightClustering, _log_cluster_stats, concat_weights, axes2perm_to_perm2axes, \
    merge_channel_clustering, NopMerge

from compression.base_clip_vit import BaseCLIPViTCompression
from compression.base_resnet import BaseResNetCompression
from compression.base_preact_resnet import BasePreActResNetCompression
from compression.base_vit import BaseViTCompression


class ResNet18_ModelFolding(BaseResNetCompression):
    def __init__(self, model, min_channels=1, compression_ratio=0.5):
        super().__init__(model, min_channels=min_channels, compression_ratio=compression_ratio)
        self.fold_entries = []
        self.grams = {}
        self._hooks = []
        self._output_merges = {}
        
    # -------- Calibration (collect Gram before folding) --------
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

    def _record_fold_entries(self, axes, merge_matrix):
        """Snapshot pre-fold consumer params plus merge matrix for compensation."""
        if merge_matrix is None:
            return
        merge_cpu = merge_matrix.detach().cpu().clone()
        recorded = set()
        for tensor_name, axis in axes:
            if axis != 1 or not tensor_name.endswith('.weight'):
                continue
            module_name, _ = tensor_name.rsplit('.', 1)
            if module_name in recorded:
                continue
            module = get_module_by_name_ResNet(self.model, module_name)
            if not isinstance(module, (nn.Conv2d, nn.Linear)):
                continue
            weight = module.weight.detach().cpu().clone()
            bias = module.bias.detach().cpu().clone() if module.bias is not None else None
            self.fold_entries.append({
                'layer_name': module_name,
                'weight_full': weight,
                'bias_full': bias,
                'original_shape': tuple(weight.shape),
                'merge_matrix': merge_cpu.clone()
            })
            recorded.add(module_name)

    def _record_output_merges(self, axes, merge_matrix):
        """Store output merge matrices for modules folded along axis 0."""
        if merge_matrix is None:
            return
        merge_cpu = merge_matrix.detach().cpu().clone()
        for tensor_name, axis in axes:
            if axis != 0 or not tensor_name.endswith('.weight'):
                continue
            module_name, _ = tensor_name.rsplit('.', 1)
            module = get_module_by_name_ResNet(self.model, module_name)
            if not isinstance(module, (nn.Conv2d, nn.Linear)):
                continue
            self._output_merges[module_name] = merge_cpu.clone()

    
    def compress_function(self, axes, params):
        """
        Folding logic: perform clustering and merge weights.
        """
        # Always cluster on output channels
        n_channels = params[axes[0][0]].shape[0]
        n_clusters = max(int(n_channels * self.keep_ratio), 1)

        # Flatten weights across layers (output dim)
        weight = concat_weights({0: axes}, params, 0, n_channels)

        # Cluster
        clusterer = WeightClustering(n_clusters=n_clusters, n_features=n_channels,
                                     method="hkmeans", normalize=False, use_pca=True)
        labels = clusterer(weight).to(self.device).long()

        # _log_cluster_stats(weight, labels, axes[0][0])

        # Build merge matrix
        merge_matrix = torch.zeros((n_clusters, n_channels), device=self.device)
        merge_matrix.scatter_(0, labels.unsqueeze(0), 1.0)
        cluster_counts = merge_matrix.sum(dim=1, keepdim=True)
        merge_matrix /= cluster_counts.clamp(min=1)  # avoid div by zero

        # Merge params
        from utils.weight_clustering import NopMerge
        compressed_params = merge_channel_clustering({0: axes}, params, 0, merge_matrix, custom_merger=NopMerge())

        return compressed_params, {
            'cluster_labels': labels,
            'merge_matrix': merge_matrix
        }

    def apply(self):
        print(f"[INFO] Starting {self.__class__.__name__}...")
        self.fold_entries = []
        self._output_merges = {}
        axis_to_perm = get_axis_to_perm_ResNet18(override=False)
        perm_to_axes = axes2perm_to_perm2axes(axis_to_perm)

        for perm_id, axes in perm_to_axes.items():
            # --- Collect raw params ---
            raw_params = {}
            for module_name, axis in axes:
                module = get_module_by_name_ResNet(self.model, module_name)
                weight = module.weight.data if hasattr(module, 'weight') else module.data
                raw_params[module_name] = weight

            # --- Call specific compression/pruning logic ---
            compressed_params, meta = self.compress_function(axes, raw_params)
            merge_matrix = meta.get('merge_matrix') if meta else None
            self._record_fold_entries(axes, merge_matrix)
            self._record_output_merges(axes, merge_matrix)

            # --- Rebuild modules ---
            param_groups = defaultdict(dict)
            for full_name, tensor in compressed_params.items():
                module_name, pname = full_name.rsplit('.', 1)
                param_groups[module_name][pname] = tensor

            for module_name, param_dict in param_groups.items():
                module = get_module_by_name_ResNet(self.model, module_name)
                cluster_labels = meta.get('cluster_labels') if meta else None
                n_clusters = param_dict['weight'].shape[0]
                new_module = self._rebuild_module(module, param_dict, cluster_labels, n_clusters)
                self._replace(module_name, new_module)

        return self.model

    def get_merge_matrix(self):
        """ get merge matrix R """
        return self.get_fold_state()

    def get_compression_state(self):
        """Return full folding metadata for downstream compensation."""
        state = []
        for entry in self.fold_entries:
            state_entry = {
                'layer_name': entry['layer_name'],
                'weight_full': entry['weight_full'].clone(),
                'bias_full': None if entry['bias_full'] is None else entry['bias_full'].clone(),
                'original_shape': entry['original_shape'],
                'merge_matrix': entry['merge_matrix'].clone()
            }
            output_merge = self._output_merges.get(entry['layer_name'])
            if output_merge is not None:
                state_entry['output_merge_matrix'] = output_merge.clone()
            state.append(state_entry)
        return state

    def get_gram_stats(self):
        """ get G in formular"""
        return self.grams


class CLIPViT_ModelFolding(BaseCLIPViTCompression):
    def __init__(self, model, min_channels=1, compression_ratio=0.5):
        super().__init__(model, min_channels, compression_ratio)
        self.fold_entries = []
        self.grams = {}
        self._hooks = []
        self._named_modules = dict(self.model.named_modules())

    # -------- Calibration: collect Gram matrices on Linear modules --------
    def _register_activation_hooks(self):
        self._grams_local = {}
        self._named_modules = dict(self.model.named_modules())

        def make_hook(name):
            def hook(mod, inputs):
                x = inputs[0]
                with torch.no_grad():
                    x2 = x.float().reshape(-1, x.shape[-1])
                    G = (x2.t() @ x2).detach().cpu()
                    self._grams_local[name] = self._grams_local.get(name, 0) + G
            return hook

        for name, module in self._named_modules.items():
            if isinstance(module, nn.Linear):
                self._hooks.append(module.register_forward_pre_hook(make_hook(name)))

    def _clear_hooks(self):
        for h in self._hooks:
            try:
                h.remove()
            except Exception:
                pass
        self._hooks = []

    @torch.no_grad()
    def run_calibration(self, dataloader, device, num_batches=50, forward_fn=None):
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
        self.grams = {name: G.to(torch.float32) for name, G in self._grams_local.items()}
        self._grams_local = {}

    def _record_fold_entry(self, layer_name, merge_matrix, weight_full, bias_full):
        if merge_matrix is None:
            return
        self.fold_entries.append({
            'layer_name': layer_name,
            'weight_full': weight_full.detach().cpu().clone(),
            'bias_full': None if bias_full is None else bias_full.detach().cpu().clone(),
            'original_shape': tuple(weight_full.shape),
            'merge_matrix': merge_matrix.detach().cpu().clone()
        })

    def compress_function(self, axes, params):
        """
        Compress weights for CLIP ViT MLP (c_fc + c_proj) using cluster means.
        Returns compressed params and the merge matrix used on hidden units.
        """
        compressed = {}

        module_fc, _ = axes[0]
        module_proj, _ = axes[1]

        W_fc = params[module_fc]  # [hidden_dim, in_dim]
        W_proj = params[module_proj]  # [out_dim, hidden_dim]

        device = W_fc.device
        dtype = W_fc.dtype

        n_channels = W_fc.shape[0]
        n_clusters = max(int(n_channels * self.keep_ratio), self.min_channels)
        n_clusters = min(n_clusters, n_channels)

        eps = torch.finfo(dtype).eps
        col_mean = W_fc.mean(dim=0, keepdim=True)
        col_std = W_fc.std(dim=0, unbiased=False, keepdim=True) + eps
        W_fc_norm = (W_fc - col_mean) / col_std

        clusterer = WeightClustering(
            n_clusters=n_clusters,
            method="hkmeans",
            use_pca=False,
            normalize=False
        )
        labels = clusterer(W_fc_norm).to(device).long()

        unique_labels = torch.unique(labels, sorted=True)
        if unique_labels.numel() < n_clusters:
            remap = {int(lbl): i for i, lbl in enumerate(unique_labels.tolist())}
            labels = torch.tensor([remap[int(l.item())] for l in labels],
                                  device=device, dtype=torch.long)
            n_clusters = unique_labels.numel()


        merge_matrix = torch.zeros((n_clusters, n_channels), device=device, dtype=dtype)
        merge_matrix.scatter_(0, labels.unsqueeze(0), 1.0)
        merge_matrix /= merge_matrix.sum(dim=1, keepdim=True).clamp(min=1)

        # --- Build de-normalized cluster means for W_fc, and sum columns for W_proj ---
        cluster_means = []
        proj_cols = []
        for cid in range(n_clusters):
            members = (labels == cid).nonzero(as_tuple=True)[0]  # row indices in this cluster

            # Mean in normalized space, then de-normalize
            mean_norm = W_fc_norm[members, :].mean(dim=0, keepdim=False)  # [in_dim]
            mean_vec = mean_norm * col_std.squeeze(0) + col_mean.squeeze(0)
            cluster_means.append(mean_vec)

            # Sum corresponding columns in W_proj (downstream combination of clustered units)
            proj_sum = W_proj[:, members].sum(dim=1, keepdim=True)  # [out_dim, 1]
            proj_cols.append(proj_sum)

        new_fc = torch.stack(cluster_means, dim=0)  # [n_clusters, in_dim]
        new_proj = torch.cat(proj_cols, dim=1)  # [out_dim, n_clusters]

        compressed[module_fc + '.weight'] = new_fc.to(device=device, dtype=dtype)
        compressed[module_proj + '.weight'] = new_proj.to(device=device, dtype=dtype)

        # --- Biases ---
        # c_fc bias: average within each cluster
        if module_fc + '.bias' in params and params[module_fc + '.bias'] is not None:
            b_fc = params[module_fc + '.bias']
            new_b = []
            for cid in range(n_clusters):
                members = (labels == cid).nonzero(as_tuple=True)[0]
                new_b.append(b_fc[members].mean())
            compressed[module_fc + '.bias'] = torch.stack(new_b, dim=0).to(device=device, dtype=dtype)

        # c_proj bias unchanged
        if module_proj + '.bias' in params and params[module_proj + '.bias'] is not None:
            compressed[module_proj + '.bias'] = params[module_proj + '.bias']

        
        # new_fc = merge_matrix @ W_fc
        # new_proj = W_proj @ merge_matrix.t()

        # compressed[module_fc + '.weight'] = new_fc.to(device=device, dtype=dtype)
        # compressed[module_proj + '.weight'] = new_proj.to(device=device, dtype=dtype)

        # if module_fc + '.bias' in params and params[module_fc + '.bias'] is not None:
        #     new_b = merge_matrix @ params[module_fc + '.bias']
        #     compressed[module_fc + '.bias'] = new_b.to(device=device, dtype=dtype)

        # if module_proj + '.bias' in params and params[module_proj + '.bias'] is not None:
        #     compressed[module_proj + '.bias'] = params[module_proj + '.bias']

        return compressed, {'merge_matrix': merge_matrix}

    def apply(self):
        print(f"[INFO] Starting {self.__class__.__name__} compression...")
        self.fold_entries = []
        axis_to_perm = get_axis_to_perm_ViT(self.model)
        embed_dim = self.model.visual.conv1.out_channels

        for group_name, axes in axis_to_perm.items():
            module_name_fc = axes[0][0]
            module_name_proj = axes[1][0]

            module_fc = get_module_by_name_ViT(self.model, module_name_fc)
            module_proj = get_module_by_name_ViT(self.model, module_name_proj)

            raw_params = {
                module_name_fc: module_fc.weight.data.clone(),
                module_name_proj: module_proj.weight.data.clone(),
                module_name_fc + '.bias': module_fc.bias.data.clone() if module_fc.bias is not None else None,
                module_name_proj + '.bias': module_proj.bias.data.clone() if module_proj.bias is not None else None,
            }

            compressed_params, meta = self.compress_function(axes, raw_params)
            merge_matrix = meta.get('merge_matrix') if meta else None
            self._record_fold_entry(
                module_name_proj,
                merge_matrix,
                raw_params[module_name_proj],
                raw_params[module_name_proj + '.bias']
            )

            param_groups = defaultdict(dict)
            for full_name, tensor in compressed_params.items():
                mod_name, pname = full_name.rsplit('.', 1)
                param_groups[mod_name][pname] = tensor

            for module_name, param_dict in param_groups.items():
                module = get_module_by_name_ViT(self.model, module_name)
                if module_name.endswith('c_fc'):
                    out_features, in_features = param_dict['weight'].shape
                elif module_name.endswith('c_proj'):
                    out_features, in_features = embed_dim, param_dict['weight'].shape[1]
                else:
                    out_features, in_features = module.out_features, module.in_features

                new_linear = nn.Linear(in_features, out_features, bias='bias' in param_dict).to(self.device)
                new_linear.weight.data.copy_(param_dict['weight'])
                if 'bias' in param_dict:
                    new_linear.bias.data.copy_(param_dict['bias'])

                parent_name = '.'.join(module_name.split('.')[:-1])
                attr_name = module_name.split('.')[-1]
                parent = get_module_by_name_ViT(self.model, parent_name) if parent_name else self.model
                setattr(parent, attr_name, new_linear)

        return self.model

    def get_compression_state(self):
        state = []
        for entry in self.fold_entries:
            state.append({
                'layer_name': entry['layer_name'],
                'weight_full': entry['weight_full'].clone(),
                'bias_full': None if entry['bias_full'] is None else entry['bias_full'].clone(),
                'original_shape': entry['original_shape'],
                'merge_matrix': entry['merge_matrix'].clone()
            })
        return state

    def get_fold_state(self):
        return self.get_compression_state()

    def get_gram_stats(self):
        return self.grams



class PreActResNet18_ModelFolding(BasePreActResNetCompression):
    def compress_function(self, axes, params):
        """
        Folding logic: perform clustering and merge weights.
        """
        n_channels = params[axes[0][0]].shape[axes[0][1]]
        n_clusters = max(int(n_channels * self.keep_ratio), 1)

        # Flatten and cluster
        weight = concat_weights({0: axes}, params, 0, n_channels)
        clusterer = WeightClustering(n_clusters=n_clusters, n_features=n_channels,
                                     method="hkmeans", normalize=False, use_pca=True)
        labels = clusterer(weight).to(self.device).long()

        # Log cluster stats
        # _log_cluster_stats(weight, labels, axes[0][0])

        # Convert to merge matrix
        merge_matrix = torch.zeros((n_clusters, n_channels), device=self.device, dtype=torch.float32)
        merge_matrix.scatter_(0, labels.unsqueeze(0), 1.0)
        merge_matrix /= merge_matrix.sum(dim=1, keepdim=True).clamp(min=1)

        # Merge weights
        from utils.weight_clustering import NopMerge
        compressed_params = merge_channel_clustering({0: axes}, params, 0, merge_matrix, custom_merger=NopMerge())

        return compressed_params, {'cluster_labels': labels}

    def apply(self):
        axis_to_perm = get_axis_to_perm_PreActResNet18(override=False)
        perm_to_axes = axes2perm_to_perm2axes(axis_to_perm)

        for perm_id, axes in perm_to_axes.items():
            # --- Collect weights ---
            features = []
            raw_params = {}
            module_offsets = {}
            offset = 0

            for module_name, axis in axes:
                module = get_module_by_name_PreActResNet18(self.model, module_name)
                weight = module.weight.data if hasattr(module, 'weight') else module.data
                raw_params[module_name] = weight
                weight = weight.transpose(0, axis).contiguous()
                n_channels = weight.shape[0]
                reshaped = weight.view(n_channels, -1)
                features.append(reshaped)
                module_offsets[module_name] = (offset, offset + n_channels)
                offset += n_channels

            # --- Cluster and fold ---
            all_features = torch.cat(features, dim=1)
            n_channels = all_features.shape[0]
            n_clusters = max(int(n_channels * self.keep_ratio), self.min_channels)

            compressed_params, merge_sizes = self.compress_function(axes, raw_params)

            # --- Rebuild modules ---
            param_groups = defaultdict(dict)
            for full_name, tensor in compressed_params.items():
                module_name, param_name = full_name.rsplit('.', 1)
                param_groups[module_name][param_name] = tensor

            for module_name, param_dict in param_groups.items():
                module = get_module_by_name_PreActResNet18(self.model, module_name)

                # Determine if this module should have BN folded
                cluster_labels = None
                if module_name in module_offsets:
                    start, end = module_offsets[module_name]
                    cluster_labels = merge_sizes.get('cluster_labels') if merge_sizes else None

                # Rebuild Conv/Linear/BN
                new_module = self._rebuild_module(
                    module_name,
                    module,
                    param_dict,
                    cluster_labels=cluster_labels,
                    n_clusters=n_clusters
                )

                # Replace in parent
                parent_name = '.'.join(module_name.split('.')[:-1])
                attr_name = module_name.split('.')[-1]
                if parent_name:
                    parent = get_module_by_name_PreActResNet18(self.model, parent_name)
                    setattr(parent, attr_name, new_module)
                else:
                    setattr(self.model, attr_name, new_module)

        print("Model folding complete (with BN folding).")
        return self.model




class ViT_ModelFolding(BaseViTCompression):
    def __init__(self, model, min_channels=1, compression_ratio=0.5):
        super().__init__(model, min_channels, compression_ratio)
        self.fold_entries = []
        self.grams = {}
        self._hooks = []
        self._named_modules = dict(self.model.named_modules())

    # -------- Calibration (Gram collection) --------
    def _register_activation_hooks(self):
        self._grams_local = {}
        self._named_modules = dict(self.model.named_modules())

        def make_hook(name):
            def hook(mod, inputs):
                x = inputs[0]
                with torch.no_grad():
                    x2 = x.float().reshape(-1, x.shape[-1])
                    G = (x2.t() @ x2).detach().cpu()
                    self._grams_local[name] = self._grams_local.get(name, 0) + G
            return hook

        for name, module in self._named_modules.items():
            if isinstance(module, nn.Linear):
                self._hooks.append(module.register_forward_pre_hook(make_hook(name)))

    def _clear_hooks(self):
        for h in self._hooks:
            try:
                h.remove()
            except Exception:
                pass
        self._hooks = []

    @torch.no_grad()
    def run_calibration(self, dataloader, device, num_batches=50):
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
        self.fold_entries = []

    def _record_fold_entry(self, proj_name, merge_matrix, weight_proj, bias_proj):
        if merge_matrix is None:
            return
        self.fold_entries.append({
            'layer_name': proj_name,
            'weight_full': weight_proj.detach().cpu().clone(),
            'bias_full': None if bias_proj is None else bias_proj.detach().cpu().clone(),
            'original_shape': tuple(weight_proj.shape),
            'merge_matrix': merge_matrix.detach().cpu().clone()
        })

    def compress_function(self, axes, params):
        compressed = {}

        module_fc, module_proj = axes
        W_fc = params[module_fc + '.weight']
        W_proj = params[module_proj + '.weight']

        device = W_fc.device
        dtype = W_fc.dtype

        n_channels = W_fc.shape[0]
        n_clusters = max(int(n_channels * self.keep_ratio), self.min_channels)
        n_clusters = min(n_clusters, n_channels)

        eps = torch.finfo(dtype).eps
        col_mean = W_fc.mean(dim=0, keepdim=True)
        col_std = W_fc.std(dim=0, unbiased=False, keepdim=True) + eps
        W_fc_norm = (W_fc - col_mean) / col_std

        clusterer = WeightClustering(
            n_clusters=n_clusters,
            method="hkmeans",
            use_pca=False,
            normalize=False
        )
        labels = clusterer(W_fc_norm).to(device).long()
        unique_labels = torch.unique(labels, sorted=True)
        if unique_labels.numel() < n_clusters:
            remap = {int(lbl): i for i, lbl in enumerate(unique_labels.tolist())}
            labels = torch.tensor([remap[int(l.item())] for l in labels], device=device, dtype=torch.long)
            n_clusters = unique_labels.numel()

        merge_matrix = torch.zeros((n_clusters, n_channels), device=device, dtype=dtype)
        merge_matrix.scatter_(0, labels.unsqueeze(0), 1.0)
        merge_matrix /= merge_matrix.sum(dim=1, keepdim=True).clamp(min=1)

        new_fc = merge_matrix @ W_fc
        new_proj = W_proj @ merge_matrix.t()

        compressed[module_fc + '.weight'] = new_fc.to(device=device, dtype=dtype)
        compressed[module_proj + '.weight'] = new_proj.to(device=device, dtype=dtype)

        if module_fc + '.bias' in params and params[module_fc + '.bias'] is not None:
            b_fc = params[module_fc + '.bias']
            new_b = merge_matrix @ b_fc
            compressed[module_fc + '.bias'] = new_b.to(device=device, dtype=dtype)

        if module_proj + '.bias' in params and params[module_proj + '.bias'] is not None:
            compressed[module_proj + '.bias'] = params[module_proj + '.bias']

        return compressed, {'merge_matrix': merge_matrix}

    def get_compression_state(self):
        state = []
        for entry in self.fold_entries:
            state.append({
                'layer_name': entry['layer_name'],
                'weight_full': entry['weight_full'].clone(),
                'bias_full': None if entry['bias_full'] is None else entry['bias_full'].clone(),
                'original_shape': entry['original_shape'],
                'merge_matrix': entry['merge_matrix'].clone()
            })
        return state

    def get_gram_stats(self):
        return self.grams
