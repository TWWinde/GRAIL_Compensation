import torch 
import torch.nn as nn 
from .layerwrapper import WrappedGPT, BiasGPT, GramRecorder
from .data import get_loaders 
from .compensation import MagnitudeCompensator
import math
from tqdm import tqdm

# create a dictionary to map the method name to the function
"""
    'IFV': Input Feature Variance
    'WIFV': Weighted Input Feature Variance
    'WIFN': Weighted Input Feature Norm
"""
metrics = {
    'IFV': lambda wrapped_layers, subset, name: wrapped_layers[name].fluc_inp,
    'WIFV': lambda wrapped_layers, subset, name: wrapped_layers[name].fluc_inp * torch.sum(subset[name].weight.data.pow(2), dim=0),
    'WIFN': lambda wrapped_layers, subset, name: (torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_inp.reshape((1,-1)))).mean(axis=0),
}


def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


def check_sparsity(model):
    """
    Check the sparsity of the weights in different layers of the model.
    
    Args:
        model (nn.Module): The model to check.
        
    Returns:
        float: Ratio of the count of non-zero weights to total parameters in the model.
    """
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    layers = model.model.layers
    intermediate_size = model.config.intermediate_size
    hidden_size = model.config.hidden_size
    
    count = 0 
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            sub_count += W.numel()
            count += W.numel()
            if 'self_attn' in name:
                total_params += hidden_size * hidden_size
                sub_params += hidden_size * hidden_size
            else:
                total_params += hidden_size * intermediate_size
                sub_params += hidden_size * intermediate_size
            if subset[name].bias is not None:
                count += subset[name].bias.data.numel()
                sub_count += subset[name].bias.data.numel()
            
        print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")

    model.config.use_cache = use_cache 
    return float(count)/total_params 


def prepare_calibration_input(model, dataloader, device, force_cpu=False):
    """
    Prepare inputs for model calibration. 
    
    Args:
        model (nn.Module): The model to prepare inputs for.
        dataloader (DataLoader): DataLoader object to fetch input data.
        device (torch.device): Device on which the model is loaded. 
        force_cpu (bool): Whether to force inputs/outputs to be stored on CPU.
        
    Returns:
        inps (torch.Tensor): Input tensor for calibration.
        outs (torch.Tensor): Output tensor for calibration.
        attention_mask (torch.Tensor): Attention mask tensor.
        position_ids (torch.Tensor): Position IDs tensor.
    """
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    if "model.embed_tokens" in getattr(model, 'hf_device_map', {}):
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    # Allocate on CPU if force_cpu is True, else on device
    storage_device = 'cpu' if force_cpu else device
    inps = torch.zeros((2048, model.seqlen, model.config.hidden_size), dtype=dtype, device=storage_device)
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp.to(storage_device) # Move to storage device
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
        
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass 
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps) # Same device as inps
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    model.config.use_cache = use_cache

    return inps, outs, attention_mask, position_ids 


def compress(layer, attn_mask, mlp_mask, attn_mean_inp, mlp_mean_inp, device, bias=True, unstr=False):
    """
    Compress a model layer by masking or pruning based on the given masks.
    
    Args:
        layer (nn.Module): The model layer to compress.
        attn_mask (torch.Tensor): The mask to apply to the attention weights.
        mlp_mask (torch.Tensor): The mask to apply to the MLP weights.
        attn_mean_inp (torch.Tensor): The mean attention input.
        mlp_mean_inp (torch.Tensor): The mean MLP input.
        device (torch.device): Device on which the model is loaded.
        bias (bool, optional): Whether to consider bias while compressing. Defaults to True.
        unstr (bool, optional): If True, only mask without real pruning. Defaults to False.
        
    Returns:
        None: This function modifies the layer in-place and doesn't return anything.
    """
    if unstr:  # Only mask, do not really prune
        # Attention Weight Masking
        if attn_mask is not None:
            retain_heads = torch.count_nonzero(attn_mask)
            attn_mask = attn_mask.repeat_interleave(128)
            # Apply the mask to the query, key and value projection weights
            layer.self_attn.q_proj.weight.data *= attn_mask.unsqueeze(-1).to(device)
            layer.self_attn.k_proj.weight.data *= attn_mask.unsqueeze(-1).to(device)
            layer.self_attn.v_proj.weight.data *= attn_mask.unsqueeze(-1).to(device)
            
            output_weight = layer.self_attn.o_proj.weight.data
            if bias:
                # Add the additional bias to compensate for the loss
                output_bias = ((attn_mean_inp * ~attn_mask.to(device)) @ output_weight.T)
                
            # Note: the weight data is masked, but the weight tensor shape remains unchanged
            if bias:
                layer.self_attn.o_proj.bias.data = output_bias
            layer.self_attn.o_proj.weight.data = output_weight

        # MLP Weight Masking
        if mlp_mask is not None:
            # Apply the mask to the up and gate projection weights
            layer.mlp.up_proj.weight.data *= mlp_mask.unsqueeze(-1).to(device)
            layer.mlp.gate_proj.weight.data *= mlp_mask.unsqueeze(-1).to(device)
            
            output_weight = layer.mlp.down_proj.weight.data
            if bias:
                # Add the additional bias to compensate for the loss
                output_bias = ((mlp_mean_inp * ~mlp_mask.to(device)) @ output_weight.T)
                
            # Note: the weight data is masked, but the weight tensor shape remains unchanged
            if bias:
                layer.mlp.down_proj.bias.data = output_bias
            layer.mlp.down_proj.weight.data = output_weight
    
    else:
        # Real Pruning
        # Attention Weight Pruning
        if attn_mask is not None:
            retain_heads = torch.count_nonzero(attn_mask)
            attn_mask = attn_mask.repeat_interleave(128)
            
            # Prune the query, key and value projection weights
            # We reduce the size of the weights based on the attention mask
            layer.self_attn.q_proj.weight.data = layer.self_attn.q_proj.weight.data[torch.where(attn_mask)[0]]
            layer.self_attn.k_proj.weight.data = layer.self_attn.k_proj.weight.data[torch.where(attn_mask)[0]]
            layer.self_attn.v_proj.weight.data = layer.self_attn.v_proj.weight.data[torch.where(attn_mask)[0]]
            
            # Update output dimensions of q, k, v projections based on remaining heads
            layer.self_attn.q_proj.out_features = attn_mask.sum().item()
            layer.self_attn.k_proj.out_features = attn_mask.sum().item()
            layer.self_attn.v_proj.out_features = attn_mask.sum().item()
            
            output_weight = layer.self_attn.o_proj.weight.data
            
            if bias:
                # Add the additional bias to compensate for the loss
                output_bias = ((attn_mean_inp * ~attn_mask.to(device)) @ output_weight.T)
                
            # Prune the output projection weight
            output_weight = layer.self_attn.o_proj.weight.data[:, torch.where(attn_mask)[0]]
            # Update layer configurations for the new output shape after pruning
            layer.self_attn.num_heads = retain_heads
            layer.self_attn.hidden_size = retain_heads * 128
            
            # Always update in_features
            layer.self_attn.o_proj.in_features = attn_mask.sum().item()
            
            if bias:
                # Re-initialize the Linear layer with new shape and bias
                # layer.self_attn.o_proj = torch.nn.Linear(in_features=output_weight.shape[1], out_features=output_weight.shape[0], bias=True).to(device)
                layer.self_attn.o_proj.bias.data = output_bias
                
            # Assign the pruned weights
            layer.self_attn.o_proj.weight.data = output_weight

        # MLP Weight Pruning
        if mlp_mask is not None:
            # Prune the up and gate projection weights
            layer.mlp.up_proj.weight.data = layer.mlp.up_proj.weight.data[torch.where(mlp_mask)[0]]
            layer.mlp.gate_proj.weight.data = layer.mlp.gate_proj.weight.data[torch.where(mlp_mask)[0]]
            
            # Update output dimensions of up and gate projections based on the mlp mask
            layer.mlp.up_proj.out_features = mlp_mask.sum().item()
            layer.mlp.gate_proj.out_features = mlp_mask.sum().item()
            
            output_weight = layer.mlp.down_proj.weight.data
            layer.mlp.intermediate_size = mlp_mask.sum().item()
            if bias:
                # Add the additional bias to compensate for the loss
                output_bias = ((mlp_mean_inp * ~mlp_mask.to(device)) @ output_weight.T)
              
            # Prune the down projection weight
            output_weight = layer.mlp.down_proj.weight.data[:, torch.where(mlp_mask)[0]]  
            
            # Always update in_features
            layer.mlp.down_proj.in_features = mlp_mask.sum().item()
            
            if bias:
                # Re-initialize the Linear layer with new shape and bias
                # layer.mlp.down_proj = torch.nn.Linear(in_features=output_weight.shape[1], out_features=output_weight.shape[0], bias=True).to(device)
                layer.mlp.down_proj.bias.data = output_bias
                
            # Assign the pruned weights
            layer.mlp.down_proj.weight.data = output_weight
        
    # Explicitly empty the CUDA cache to clean up some memory
    torch.cuda.empty_cache()
    
    
def cal_remove_neuron(args, model):
    intermediate_size = model.config.intermediate_size
    hidden_size = model.config.hidden_size
    num_layers = model.config.num_hidden_layers
    if args.structure == "UL-MM":
        remove_params = args.pruning_ratio * (intermediate_size * hidden_size * 3 + hidden_size * hidden_size * 4)
        remove_head_params = hidden_size * 4 * (args.remove_heads // num_layers) * 128
        return int((remove_params - remove_head_params) / (hidden_size * 3))
    else:
        remove_params = num_layers * args.pruning_ratio * (intermediate_size * hidden_size * 3 + hidden_size * hidden_size * 4)
        remove_head_params = hidden_size * 4 * args.remove_heads * 128
        return int((remove_params - remove_head_params) / (hidden_size * 3))


def prune_flap(args, model, tokenizer, device=torch.device("cuda:0")):
    """
    Our FLAP Pruning.
    
    Args:
        args (object): Command line arguments parsed via argparse.
        model (nn.Module): PyTorch model to prune.
        tokenizer (Tokenizer): Tokenizer associated with the model.
        device (torch.device, optional): Device to move tensors to. Defaults to CUDA device 0.
    """
    use_cache = model.config.use_cache 
    model.config.use_cache = False 
    
    print(f"loading calibration data from {args.dataset}")
    dataloader, _ = get_loaders(args.dataset, nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    print("dataset loading complete")
    
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device, force_cpu=args.compensate)
    layers = model.model.layers

    attn_metric_list, mlp_metric_list = [], []
    attn_baseline_inp_list, mlp_baseline_inp_list = [], []
    attn_mask, mlp_mask = [], []
        
    # Split into sub-problems, separate statistics for each module
    for i in tqdm(range(len(layers)), desc="Processing layers"):
        layer = layers[i]
        subset = {}
        subset.update({'self_attn.o_proj': find_layers(layer)['self_attn.o_proj']})
        subset.update({'mlp.down_proj': find_layers(layer)['mlp.down_proj']})

        if f"model.layers.{i}" in getattr(model, 'hf_device_map', {}):   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
        else:
            dev = device

        wrapped_layers = {}
        for name in subset:
                wrapped_layers[name] = BiasGPT(subset[name], args.metrics)            

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                # Move input batch to device, run layer, move output back to CPU
                layer_input = inps[j].unsqueeze(0).to(dev)
                layer_out = layer(layer_input, attention_mask=attention_mask.to(dev), position_ids=position_ids.to(dev))[0]
                if inps.device.type == 'cpu':
                    outs[j] = layer_out.cpu()
                else:
                    outs[j] = layer_out
        for h in handles:
            h.remove()

        for name in subset:
            if name == 'self_attn.o_proj':
                W_metric = metrics[args.metrics](wrapped_layers, subset, name) ** 2
                if args.structure == "UL-UM":
                    W_metric = W_metric.reshape(-1, 128).sum(dim=1)
                    thresh = torch.sort(W_metric.cuda())[0][int(args.pruning_ratio*layer.self_attn.num_heads)].cpu()
                    W_mask = (W_metric>=thresh)
                    attn_mask.append(W_mask)
                elif args.structure == "UL-MM":
                    W_metric = W_metric.reshape(-1, 128).sum(dim=1)
                    thresh = torch.sort(W_metric.cuda())[0][args.remove_heads // len(layers)].cpu()
                    W_mask = (W_metric>=thresh)
                    attn_mask.append(W_mask)
                else:
                    attn_metric_list.append(W_metric.cpu())
                attn_baseline_inp_list.append(wrapped_layers[name].baseline_inp.type(torch.half))
            else:
                W_metric = metrics[args.metrics](wrapped_layers, subset, name)
                if args.structure == "UL-UM":
                    thresh = torch.sort(W_metric.cuda())[0][int(W_metric.numel()*args.pruning_ratio)].cpu()
                    W_mask = (W_metric>=thresh)
                    mlp_mask.append(W_mask)
                elif args.structure == "UL-MM":
                    thresh = torch.sort(W_metric.cuda())[0][cal_remove_neuron(args, model)].cpu()
                    W_mask = (W_metric>=thresh)
                    mlp_mask.append(W_mask)
                else:
                    mlp_metric_list.append(W_metric.cpu())
                mlp_baseline_inp_list.append(wrapped_layers[name].baseline_inp.type(torch.half))
            
            wrapped_layers[name].free()

        inps, outs = outs, inps # Use the original output as input to the next layer
        torch.cuda.empty_cache()

    standarlization = lambda x: (x - torch.mean(x, axis=1, keepdim=True)) / torch.std(x, axis=1, keepdim=True)

    if args.structure in ["AL-MM", "AL-AM"]:
        attn_metric = torch.stack(attn_metric_list)
        attn_metric = standarlization(attn_metric)
        attn_metric = attn_metric.reshape(len(layers), -1, 128).mean(dim=2)
        
        mlp_metric = torch.stack(mlp_metric_list)
        mlp_metric = standarlization(mlp_metric)
        
        if args.structure == "AL-MM":
            sorted_attn = torch.sort(attn_metric.view(-1), descending=True)[0]
            attn_thres = sorted_attn[-int(args.remove_heads)]
            attn_mask = (attn_metric > attn_thres)  # 1 means retain
            
            sorted_mlp = torch.sort(mlp_metric.view(-1), descending=True)[0]
            mlp_thres = sorted_mlp[-cal_remove_neuron(args, model)]
            mlp_mask = (mlp_metric > mlp_thres)
        else:
            prune_metric = torch.cat([attn_metric.view(-1), mlp_metric.view(-1)])
            sorted_prune, indices = torch.sort(prune_metric, descending=True)
            compression_weight = torch.ones_like(indices)
            compression_weight[indices < attn_metric.numel()] = 512.0 / 3
            threshold = sorted_prune[torch.argmin(torch.abs(torch.cumsum(compression_weight, 0) - torch.sum(compression_weight)*(1 - args.pruning_ratio)))]
            attn_mask = (attn_metric > threshold)
            mlp_mask = (mlp_metric > threshold)
    else:
        attn_mask = torch.stack(attn_mask) 
        mlp_mask = torch.stack(mlp_mask)
    
    if args.compensate:
        # --- Sequential Pruning & Compensation ---
        print("Resetting inputs for sequential pruning & compensation...")
        # Clean up memory from the first pass before re-allocating
        del inps, outs
        torch.cuda.empty_cache()
        
        with torch.no_grad():
            inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device, force_cpu=True)

        use_bias = not args.no_bias_compensation # Allow bias compensation with weight compensation if not explicitly disabled

        for i in tqdm(range(len(layers)), desc="Sequential Pruning & Compensation"):
            layer = layers[i]
            subset = {}
            subset.update({'self_attn.o_proj': find_layers(layer)['self_attn.o_proj']})
            subset.update({'mlp.down_proj': find_layers(layer)['mlp.down_proj']})

            if f"model.layers.{i}" in getattr(model, 'hf_device_map', {}): 
                dev = model.hf_device_map[f"model.layers.{i}"]
            else:
                dev = device

            # 1. Collect Gram Matrices
            gram_recorders = {}
            for name in subset:
                gram_recorders[name] = GramRecorder(subset[name])
            
            def add_batch_gram(name):
                def tmp(_, inp, out):
                    gram_recorders[name].add_batch(inp[0].data, out.data)
                return tmp

            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch_gram(name)))
            
            # Use comp_nsamples for Gram matrix collection (compensation)
            comp_nsamples = getattr(args, 'comp_nsamples', args.nsamples)
            for j in range(comp_nsamples):
                with torch.no_grad():
                    # Move input to device just for Gram collection
                    layer_input = inps[j].unsqueeze(0).to(dev)
                    layer(layer_input, attention_mask=attention_mask.to(dev), position_ids=position_ids.to(dev))
            
            for h in handles:
                h.remove()

            # 2. Prepare Compensation Entries (Save Original Weights)
            entries = []
            # Attention
            name_attn = 'self_attn.o_proj'
            if name_attn in gram_recorders:
                feature_mask = attn_mask[i].repeat_interleave(128).to(dev)
                entry = {
                    'name': f"layer_{i}_{name_attn}",
                    'module': layer.self_attn.o_proj,
                    'weight_full': layer.self_attn.o_proj.weight.data.clone(), 
                    'mask': ~feature_mask if args.unstr else None,
                    'in_keep': torch.where(feature_mask)[0] if not args.unstr else None,
                    'mean_input': attn_baseline_inp_list[i].to(dev).float()
                }
                entries.append(entry)
            
            # MLP
            name_mlp = 'mlp.down_proj'
            if name_mlp in gram_recorders:
                entry = {
                    'name': f"layer_{i}_{name_mlp}",
                    'module': layer.mlp.down_proj,
                    'weight_full': layer.mlp.down_proj.weight.data.clone(),
                    'mask': ~mlp_mask[i].to(dev) if args.unstr else None,
                    'in_keep': torch.where(mlp_mask[i].to(dev))[0] if not args.unstr else None,
                    'mean_input': mlp_baseline_inp_list[i].to(dev).float()
                }
                entries.append(entry)

            # 3. Compress (Prune)
            if f"model.layers.{i}" in getattr(model, 'hf_device_map', {}): 
                compress(layer, attn_mask[i], None, attn_baseline_inp_list[i], None, dev, bias=use_bias, unstr=args.unstr)
                compress(layer, None, mlp_mask[i], None, mlp_baseline_inp_list[i], dev, bias=use_bias, unstr=args.unstr)
            else:
                compress(layer, attn_mask[i], None, attn_baseline_inp_list[i], None, dev, bias=use_bias, unstr=args.unstr)
                compress(layer, None, mlp_mask[i], None, mlp_baseline_inp_list[i], dev, bias=use_bias, unstr=args.unstr)

            # 4. Compensate
            if entries:
                compensator = MagnitudeCompensator(model, ridge_lambda=args.ridge_lambda)
                grams = {}
                for e in entries:
                    short_name = e['name'].split(f"layer_{i}_")[-1]
                    grams[e['name']] = gram_recorders[short_name].gram
                    gram_recorders[short_name].free()
                
                compensator.load_gram_stats(grams)
                compensator.load_compression_state(entries)
                compensator.compensate()
                
                del grams
                del entries
                del compensator
            
            # 5. Update inputs for next layer
            for j in range(args.nsamples):
                with torch.no_grad():
                    # Move input to device, run layer (now pruned+compensated), move output back to CPU
                    layer_input = inps[j].unsqueeze(0).to(dev)
                    layer_out = layer(layer_input, attention_mask=attention_mask.to(dev), position_ids=position_ids.to(dev))[0]
                    outs[j] = layer_out.cpu()
            
            inps, outs = outs, inps
            torch.cuda.empty_cache()
            
    else:
        # --- Original FLAP Pruning (No Compensation) ---
        use_bias = not args.no_bias_compensation
        for idx in range(len(layers)):
            if f"model.layers.{idx}" in getattr(model, 'hf_device_map', {}): 
                dev = model.hf_device_map[f"model.layers.{idx}"]
            else:
                dev = device
                
            compress(model.model.layers[idx], attn_mask[idx], None, attn_baseline_inp_list[idx], None, dev, bias=use_bias, unstr=args.unstr)
            compress(model.model.layers[idx], None, mlp_mask[idx], None, mlp_baseline_inp_list[idx], dev, bias=use_bias, unstr=args.unstr)

    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()
   
   
def prune_wanda_sp(args, model, tokenizer, device=torch.device("cuda:0")):
    """
    Wanda on structured pruning.

    Args:
        args (object): Command line arguments parsed via argparse.
        model (nn.Module): PyTorch model to prune.
        tokenizer (Tokenizer): Tokenizer associated with the model.
        device (torch.device, optional): Device to move tensors to. Defaults to CUDA device 0.
    """
    use_cache = model.config.use_cache 
    model.config.use_cache = False 
    
    print(f"loading calibration data from {args.dataset}")
    dataloader, _ = get_loaders(args.dataset, nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    print("dataset loading complete")
    
    # Use force_cpu to avoid OOM if compensating
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device, force_cpu=args.compensate)

    # Wanda does not support FLAP-style bias compensation as it doesn't collect mean inputs
    use_bias = False
    layers = model.model.layers
    
    for i in range(len(layers)):
        layer = layers[i]
        subset = {}
        subset.update({'self_attn.o_proj': find_layers(layer)['self_attn.o_proj']})
        subset.update({'mlp.down_proj': find_layers(layer)['mlp.down_proj']})

        if f"model.layers.{i}" in getattr(model, 'hf_device_map', {}): 
            dev = model.hf_device_map[f"model.layers.{i}"]
        else:
            dev = device

        wrapped_layers = {}
        gram_recorders = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])
            if args.compensate:
                gram_recorders[name] = GramRecorder(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
                if args.compensate:
                    gram_recorders[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in subset:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        
        # Use comp_nsamples for Gram matrix collection when compensating
        loop_nsamples = getattr(args, 'comp_nsamples', args.nsamples) if args.compensate else args.nsamples
        for j in range(loop_nsamples):
            with torch.no_grad():
                # Handle CPU/GPU inputs
                layer_input = inps[j].unsqueeze(0).to(dev)
                layer(layer_input, attention_mask=attention_mask.to(dev), position_ids=position_ids.to(dev))
                
        for h in handles:
            h.remove()

        entries = []
        for name in subset:
            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
            
            if name == 'self_attn.o_proj':
                W_metric = W_metric.mean(axis=0).reshape(-1, 128).sum(dim=1)    # importance score of each head
                thresh = torch.sort(W_metric.cuda())[0][int(args.pruning_ratio*layer.self_attn.num_heads)].cpu()
                W_mask = (W_metric>=thresh)
            else:
                W_metric = W_metric.mean(axis=0)
                thresh = torch.sort(W_metric.cuda())[0][int(W_metric.numel()*args.pruning_ratio)].cpu()
                W_mask = (W_metric>=thresh)
            
            # Save Original Weights BEFORE Pruning (Critical for Compensation)
            if args.compensate and name in gram_recorders:
                if name == 'self_attn.o_proj':
                     mask_expanded = W_mask.repeat_interleave(128).to(dev)
                else:
                     mask_expanded = W_mask.to(dev)
                     
                entry = {
                    'name': f"layer_{i}_{name}",
                    'module': subset[name],
                    'weight_full': subset[name].weight.data.clone(), # Save ORIGINAL weights
                    'mask': ~mask_expanded if args.unstr else None,
                    'in_keep': torch.where(mask_expanded)[0] if not args.unstr else None
                }
                entries.append(entry)

            # Prune
            if name == 'self_attn.o_proj':
                compress(layer, W_mask, None, None, None, dev, bias=use_bias, unstr=args.unstr)
            else:
                compress(layer, None, W_mask, None, None, dev, bias=use_bias, unstr=args.unstr)
          
            wrapped_layers[name].free()

        # Compensate (Per Layer)
        if args.compensate and entries:
            print(f"Compensating layer {i}...")
            compensator = MagnitudeCompensator(model, ridge_lambda=args.ridge_lambda)
            grams = {}
            for e in entries:
                short_name = e['name'].split(f"layer_{i}_")[-1]
                grams[e['name']] = gram_recorders[short_name].gram
                gram_recorders[short_name].free()
            
            compensator.load_gram_stats(grams)
            compensator.load_compression_state(entries)
            compensator.compensate()
            
            del grams
            del entries
            del compensator

        # Update Outputs
        for j in range(args.nsamples):
            with torch.no_grad():
                layer_input = inps[j].unsqueeze(0).to(dev)
                layer_out = layer(layer_input, attention_mask=attention_mask.to(dev), position_ids=position_ids.to(dev))[0]
                if inps.device.type == 'cpu':
                    outs[j] = layer_out.cpu()
                else:
                    outs[j] = layer_out
                    
        inps, outs = outs, inps # the pruned output as input to the next layer
        
        torch.cuda.empty_cache()

    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()
    
    
def prune_magnitude_sp(args, model, tokenizer, device=torch.device("cuda:0")):
    """
    Magnitude Pruning on structured pruning.
    
    Args:
        args (object): Command line arguments parsed via argparse.
        model (nn.Module): PyTorch model to prune.
        tokenizer (Tokenizer): Tokenizer associated with the model.
        device (torch.device, optional): Device to move tensors to. Defaults to CUDA device 0.
    """
    layers = model.model.layers 

    if args.compensate:
        print("loading calibration data for compensation")
        dataloader, _ = get_loaders("c4",nsamples=128,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
        with torch.no_grad():
            inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)

    for i in range(len(layers)):
        layer = layers[i]
        subset = {}
        subset.update({'self_attn.o_proj': find_layers(layer)['self_attn.o_proj']})
        subset.update({'mlp.down_proj': find_layers(layer)['mlp.down_proj']})

        gram_recorders = {}
        if args.compensate:
            if f"model.layers.{i}" in getattr(model, 'hf_device_map', {}):
                dev = model.hf_device_map[f"model.layers.{i}"]
                inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)
            
            for name in subset:
                gram_recorders[name] = GramRecorder(subset[name])
                
            def add_batch(name):
                def tmp(_, inp, out):
                    gram_recorders[name].add_batch(inp[0].data, out.data)
                return tmp

            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            
            for j in range(128):
                with torch.no_grad():
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            
            for h in handles:
                h.remove()

        for name in subset:
            print(f"pruning layer {i} name {name}")
            W_metric = torch.norm(subset[name].weight.data, dim=0)

            if name == 'self_attn.o_proj':
                W_metric = W_metric.reshape(-1, 128).sum(dim=1) # importance score of each head
                thresh = torch.sort(W_metric.cuda())[0][int(args.pruning_ratio*layer.self_attn.num_heads)].cpu()
                W_mask = (W_metric>=thresh)
                compress(layer, W_mask, None, None, None, device, bias=False, unstr=args.unstr)
            else:
                thresh = torch.sort(W_metric.cuda())[0][int(W_metric.numel()*args.pruning_ratio)].cpu()
                W_mask = (W_metric>=thresh)
                compress(layer, None, W_mask, None, None, device, bias=False, unstr=args.unstr)
            
            if args.compensate:
                print(f"Compensating layer {i} name {name}")
                compensator = MagnitudeCompensator(model, ridge_lambda=args.ridge_lambda)
                entry = {
                    'name': f"layer_{i}_{name}",
                    'module': subset[name],
                    'weight_full': subset[name].weight.data.clone(),
                    'mask': ~W_mask.to(device) if args.unstr else None,
                    'in_keep': torch.where(W_mask.to(device))[0] if not args.unstr else None
                }
                grams = {f"layer_{i}_{name}": gram_recorders[name].gram}
                compensator.load_gram_stats(grams)
                compensator.load_compression_state([entry])
                compensator.compensate()
                gram_recorders[name].free()

        if args.compensate:
            inps, outs = outs, inps
            torch.cuda.empty_cache()
            
def prune_slimgpt(args, model, tokenizer, device=torch.device("cuda:0")):
    """
    SlimGPT Pruning: Calibration -> Structured OBS Pruning -> Compensation
    """
    print("🚀 Starting SlimGPT Pruning...")
    
    use_cache = model.config.use_cache
    model.config.use_cache = False
    
    print(f"loading calibration data from {args.dataset}")
    dataloader, _ = get_loaders(args.dataset, nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    print("dataset loading complete")

    # Use force_cpu to avoid OOM if compensating
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device, force_cpu=args.compensate)

    layers = model.model.layers
    
    for i in range(len(layers)):
        print(f"📍 Processing layer {i}/{len(layers)}")
        layer = layers[i]
        
        if f"model.layers.{i}" in getattr(model, 'hf_device_map', {}): 
            dev = model.hf_device_map[f"model.layers.{i}"]
        else:
            dev = device

        subset = {
            'self_attn.o_proj': layer.self_attn.o_proj,
            'mlp.down_proj': layer.mlp.down_proj
        }

        gram_recorders = {}
        for name in subset:
            gram_recorders[name] = GramRecorder(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                gram_recorders[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in subset:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
            
        # Use comp_nsamples for Gram matrix collection when compensating
        loop_nsamples = getattr(args, 'comp_nsamples', args.nsamples) if args.compensate else args.nsamples
        for j in range(loop_nsamples):
            with torch.no_grad():
                layer_input = inps[j].unsqueeze(0).to(dev)
                layer(layer_input, attention_mask=attention_mask.to(dev), position_ids=position_ids.to(dev))
                
        for h in handles:
            h.remove()

        entries = []
        for name in subset:
            print(f"   Pruning {name}")
            weight = subset[name].weight.data
            H = gram_recorders[name].gram.float()
            
            # Add damping
            damp = 0.01 * torch.mean(torch.diag(H))
            H_inv = torch.linalg.inv(H + damp * torch.eye(H.shape[0], device=dev))

            if name == 'self_attn.o_proj':
                # Head-wise scoring
                num_heads = layer.self_attn.config.num_attention_heads
                head_dim = layer.self_attn.config.hidden_size // num_heads
                scores = []
                for h in range(num_heads):
                    start = h * head_dim
                    end = (h + 1) * head_dim
                    W_g = weight[:, start:end]
                    H_inv_gg = H_inv[start:end, start:end]
                    try:
                        H_inv_gg_inv = torch.linalg.inv(H_inv_gg)
                    except RuntimeError:
                        H_inv_gg_inv = torch.eye(head_dim, device=dev)
                    
                    term1 = W_g.T @ W_g
                    score_h = torch.sum(term1 * H_inv_gg_inv)
                    scores.append(score_h)
                
                W_metric = torch.stack(scores)
                thresh = torch.sort(W_metric)[0][int(args.pruning_ratio * num_heads)]
                W_mask = (W_metric >= thresh)
                
                feature_mask = W_mask.repeat_interleave(head_dim)
                
            else:
                # Neuron-wise scoring
                W_sq_norm = torch.sum(weight ** 2, dim=0)
                H_inv_diag = torch.diag(H_inv)
                W_metric = W_sq_norm / (H_inv_diag + 1e-6)
                
                thresh = torch.sort(W_metric)[0][int(W_metric.numel() * args.pruning_ratio)]
                W_mask = (W_metric >= thresh)
                feature_mask = W_mask

            # Save Original Weights BEFORE Pruning (Critical for Compensation)
            if args.compensate:
                entry = {
                    'name': f"layer_{i}_{name}",
                    'module': subset[name],
                    'weight_full': subset[name].weight.data.clone(), 
                    'mask': ~feature_mask.to(dev) if args.unstr else None,
                    'in_keep': torch.where(feature_mask.to(dev))[0] if not args.unstr else None
                }
                entries.append(entry)

            # Prune
            if name == 'self_attn.o_proj':
                compress(layer, W_mask, None, None, None, dev, bias=False, unstr=args.unstr)
            else:
                compress(layer, None, W_mask, None, None, dev, bias=False, unstr=args.unstr)
        
        # Compensate (Per Layer)
        if args.compensate and entries:
            print(f"Compensating layer {i}...")
            compensator = MagnitudeCompensator(model, ridge_lambda=args.ridge_lambda)
            grams = {}
            for e in entries:
                short_name = e['name'].split(f"layer_{i}_")[-1]
                grams[e['name']] = gram_recorders[short_name].gram
                gram_recorders[short_name].free()
            
            compensator.load_gram_stats(grams)
            compensator.load_compression_state(entries)
            compensator.compensate()
            
            del grams
            del entries
            del compensator
        else:
             for name in gram_recorders:
                gram_recorders[name].free()

        # Update Outputs
        for j in range(args.nsamples):
            with torch.no_grad():
                layer_input = inps[j].unsqueeze(0).to(dev)
                layer_out = layer(layer_input, attention_mask=attention_mask.to(dev), position_ids=position_ids.to(dev))[0]
                if inps.device.type == 'cpu':
                    outs[j] = layer_out.cpu()
                else:
                    outs[j] = layer_out
                    
        inps, outs = outs, inps 
        torch.cuda.empty_cache()

    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()

def prune_wanda_pp_sp(args, model, tokenizer, device=torch.device("cuda:0")):
    """
    Wanda++ on structured pruning.
    Combines Regional Gradient Score (RGS) pruning with Regional Optimization (RO).
    
    Args:
        args (object): Command line arguments parsed via argparse.
        model (nn.Module): PyTorch model to prune.
        tokenizer (Tokenizer): Tokenizer associated with the model.
        device (torch.device, optional): Device to move tensors to. Defaults to CUDA device 0.
    """
    from torch.optim import Adam
    
    use_cache = model.config.use_cache 
    model.config.use_cache = False 
    
    print(f"loading calibration data from {args.dataset}")
    dataloader, _ = get_loaders(args.dataset, nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    print("dataset loading complete")
    
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device, force_cpu=True)

    layers = model.model.layers
    
    for i in range(len(layers)):
        print(f"Processing layer {i} with Wanda++...")
        layer = layers[i]
        subset = {}
        subset.update({'self_attn.o_proj': find_layers(layer)['self_attn.o_proj']})
        subset.update({'mlp.down_proj': find_layers(layer)['mlp.down_proj']})

        if f"model.layers.{i}" in getattr(model, 'hf_device_map', {}): 
            dev = model.hf_device_map[f"model.layers.{i}"]
        else:
            dev = device

        # CRITICAL: Convert entire layer to FP32 for numerical stability
        # Save original dtype to restore later
        original_dtype = next(layer.parameters()).dtype
        layer = layer.float()  # Convert to FP32
        print(f"  Layer converted to FP32 for stability (from {original_dtype})")

        wrapped_layers = {}
        gram_recorders = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])
            if args.compensate:
                gram_recorders[name] = GramRecorder(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
                if args.compensate:
                    gram_recorders[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in subset:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
            
        # 1. Collect Input Statistics (Wanda component)
        # Use comp_nsamples for Gram matrix collection when compensating
        print(f"  Collecting input stats for layer {i}...")
        loop_nsamples = getattr(args, 'comp_nsamples', args.nsamples) if args.compensate else args.nsamples
        for j in range(loop_nsamples):
            with torch.no_grad():
                # Use FP32 for numerical stability
                layer_input = inps[j].unsqueeze(0).to(dev).float()
                layer(layer_input, attention_mask=attention_mask.to(dev), position_ids=position_ids.to(dev))
                
        for h in handles:
            h.remove()
            
        # Save Original Weights for Compensation
        entries = []
        if args.compensate:
            for name in subset:
                entries.append({
                    'name': f"layer_{i}_{name}",
                    'module': subset[name],
                    'weight_full': subset[name].weight.data.clone(),
                    # Mask will be filled later
                })
            
        # 2. Compute Regional Gradient Score (RGS) according to paper Eq. (4)
        print(f"  Computing RGS gradients for layer {i}...")
        
        # Enable gradients for this block's parameters
        for name in subset:
            subset[name].weight.requires_grad = True
         
        # According to paper Appendix A and Eq. (3):
        # G_ij = sqrt(Σ_n (∇L_l_RGS(X_l_n))^2 / N)
        # We need to accumulate squared gradients, then take sqrt(mean)
        accumulated_sq_grads = {name: torch.zeros_like(subset[name].weight) for name in subset}

        for name in subset:
            scaler_row = wrapped_layers[name].scaler_row
            if torch.isnan(scaler_row).any() or torch.isinf(scaler_row).any():
                print(f"    Warning: scaler_row for {name} contains NaN/Inf. Replacing with small values.")
                eps = 1e-8
                wrapped_layers[name].scaler_row = torch.nan_to_num(scaler_row, nan=eps, posinf=1e4, neginf=eps)
        
        for j in range(args.nsamples):
            layer.zero_grad()  # Clear previous gradients
            
            # Use FP32 for stability during gradient computation
            layer_input = inps[j].unsqueeze(0).to(dev).float()
            layer_input.requires_grad = False
            
            # Forward pass with gradient enabled (compute in FP32)
            with torch.amp.autocast(device_type='cuda', enabled=False):
                out = layer(layer_input, attention_mask=attention_mask.to(dev), position_ids=position_ids.to(dev))[0]
                out = out.float()  # Ensure output is FP32
            
            # Loss: L = ||f(X)||_2 as in paper, normalize by sqrt(numel) to prevent overflow
            out_norm = torch.norm(out, p=2) / (out.numel() ** 0.5)  
            
            eps = 1e-8
            loss = out_norm + eps  # Add epsilon to prevent zero norm
            
            # Backward to compute gradients
            loss.backward()
            
            # Accumulate SQUARED gradients (not absolute values)
            for name in subset:
                if subset[name].weight.grad is not None:
                    # Check for NaN/Inf in gradients before accumulating
                    grad = subset[name].weight.grad
                    
                    grad_is_invalid = torch.isnan(grad).any() or torch.isinf(grad).any()
                    grad_is_large = (grad.abs() > 1e4).any()
                    
                    if grad_is_invalid or grad_is_large:
                        if grad_is_invalid:
                            print(f"    Warning: Gradient for {name} contains NaN/Inf at sample {j}. Using small random gradient.")
                        else:
                            print(f"    Warning: Gradient for {name} contains large values (>1e4) at sample {j}. Clipping.")
                        
                        grad = torch.randn_like(grad) * 1e-8
                    
                    grad_norm = torch.norm(grad)
                    if grad_norm > 100:
                        print(f"    Warning: Gradient norm {grad_norm:.2e} too large at sample {j}. Clipping.")
                        grad = grad * (100 / grad_norm)
                    
                    accumulated_sq_grads[name] += grad ** 2
        
        # Compute RMS gradients: sqrt(mean(squared_gradients))
        # Add small epsilon to avoid sqrt(0) which can cause NaN
        eps = 1e-8
        for name in subset:
            if accumulated_sq_grads[name].sum() > 0:
                # G = sqrt(Σ gradient^2 / N) as in paper Eq. (3)
                G_rms = torch.sqrt(accumulated_sq_grads[name] / args.nsamples + eps)
                
                # Store in .grad for compatibility with later code
                subset[name].weight.grad = G_rms
                
                # Check for NaN/Inf after sqrt
                if torch.isnan(G_rms).any() or torch.isinf(G_rms).any():
                    print(f"    Warning: RMS gradient for {name} contains NaN/Inf after sqrt. Clamping.")
                    subset[name].weight.grad = torch.nan_to_num(G_rms, nan=0.0, posinf=1e4, neginf=-1e4)
            else:
                print(f"    Warning: No gradient accumulated for {name}, using zeros.")
                subset[name].weight.grad = torch.zeros_like(subset[name].weight)

        # 3. Compute Dense Targets for RO
        # We need the output of the dense block for all calibration samples to use as targets in RO.
        # We do this BEFORE pruning.
        print(f"  Computing dense targets for layer {i}...")
        
        # DEBUG: Print weight shapes BEFORE pruning
        print(f"    DEBUG BEFORE PRUNE: o_proj weight shape = {layer.self_attn.o_proj.weight.shape}")
        print(f"    DEBUG BEFORE PRUNE: down_proj weight shape = {layer.mlp.down_proj.weight.shape}")
        
        targets = []
        with torch.no_grad():
            for j in range(args.nsamples):
                # Use FP32 for numerical stability
                layer_input = inps[j].unsqueeze(0).to(dev).float()
                out = layer(layer_input, attention_mask=attention_mask.to(dev), position_ids=position_ids.to(dev))[0]
                targets.append(out.float().cpu()) # Store on CPU to save memory, ensure FP32
        
        # DEBUG: Check first target stats
        print(f"    DEBUG: First target mean={targets[0].mean().item():.6f}, std={targets[0].std().item():.6f}")

        # 4. Calculate Score & Prune
        print(f"  Pruning layer {i}...")
        masks = {} # Store masks for RO
        
        for name in subset:
            # G_ij: Gradient Magnitude
            if subset[name].weight.grad is not None:
                G = torch.abs(subset[name].weight.grad)
            else:
                G = torch.zeros_like(subset[name].weight.data)
            
            # ||X_j||_2: Input Norm (from WrappedGPT)
            # wrapped_layers[name].scaler_row is mean(x^2)
            # We use sqrt(scaler_row) as proxy for ||X||_2
            # Add epsilon to avoid sqrt(0) which can cause NaN in gradients
            eps = 1e-8
            X_norm = torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)) + eps)
            
            # W_ij: Weight Magnitude
            W = torch.abs(subset[name].weight.data)
            
            # Score: S = (alpha * G + X_norm) * W
            # Ensure shapes match for broadcasting
            # G: [out, in], X_norm: [1, in], W: [out, in]
            S = (args.alpha * G + X_norm) * W
            
            # Check for NaN/Inf in score
            if torch.isnan(S).any() or torch.isinf(S).any():
                print(f"    Warning: Score for {name} contains NaN/Inf. Replacing with zeros.")
                S = torch.nan_to_num(S, nan=0.0, posinf=1e4, neginf=-1e4)
            
            # Determine Mask
            if name == 'self_attn.o_proj':
                W_metric = S.mean(axis=0).reshape(-1, 128).sum(dim=1) # Head importance
                thresh = torch.sort(W_metric.cuda())[0][int(args.pruning_ratio*layer.self_attn.num_heads)].cpu()
                W_mask = (W_metric>=thresh)
                compress(layer, W_mask, None, None, None, dev, bias=False, unstr=args.unstr)
                
                # Store expanded mask for RO and Compensation
                mask_expanded = W_mask.repeat_interleave(128).to(dev)
                masks[name] = mask_expanded
                
                if args.compensate:
                    # Find entry for this layer
                    for e in entries:
                        if e['name'] == f"layer_{i}_{name}":
                            e['mask'] = ~mask_expanded if args.unstr else None
                            e['in_keep'] = torch.where(mask_expanded)[0] if not args.unstr else None
                            break
                
            else:
                W_metric = S.mean(axis=0) # Neuron importance
                thresh = torch.sort(W_metric.cuda())[0][int(W_metric.numel()*args.pruning_ratio)].cpu()
                W_mask = (W_metric>=thresh)
                compress(layer, None, W_mask, None, None, dev, bias=False, unstr=args.unstr)
                
                masks[name] = W_mask.to(dev)
                
                if args.compensate:
                    # Find entry for this layer
                    for e in entries:
                        if e['name'] == f"layer_{i}_{name}":
                            e['mask'] = ~W_mask.to(dev) if args.unstr else None
                            e['in_keep'] = torch.where(W_mask.to(dev))[0] if not args.unstr else None
                            break
            
            wrapped_layers[name].free()

        # FIX: Re-wrap parameters to update shape metadata for Autograd
        # This is necessary because compress() modifies .data in-place, which can confuse autograd
        if 'self_attn.o_proj' in masks:
            for mod in [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj, layer.self_attn.o_proj]:
                mod.weight = nn.Parameter(mod.weight.data)
        
        if 'mlp.down_proj' in masks:
             for mod in [layer.mlp.up_proj, layer.mlp.gate_proj, layer.mlp.down_proj]:
                mod.weight = nn.Parameter(mod.weight.data)

        # DEBUG: Print weight shapes AFTER pruning
        print(f"    DEBUG AFTER PRUNE: o_proj weight shape = {layer.self_attn.o_proj.weight.shape}")
        print(f"    DEBUG AFTER PRUNE: down_proj weight shape = {layer.mlp.down_proj.weight.shape}")

        # 5a. Compensation BEFORE RO (if compensate_first is set)
        if args.compensate and getattr(args, 'compensate_first', False):
            print(f"  Compensating layer {i} (before RO)...")
            compensator = MagnitudeCompensator(model, ridge_lambda=args.ridge_lambda)
            grams = {}
            for e in entries:
                short_name = e['name'].split(f"layer_{i}_")[-1]
                grams[e['name']] = gram_recorders[short_name].gram
                # Don't free gram_recorders yet if we might need them for RO targets recomputation
            
            compensator.load_gram_stats(grams)
            compensator.load_compression_state(entries)
            compensator.compensate()
            
            del grams
            del compensator
            
            # NOTE: We do NOT recompute targets after compensation!
            # RO should optimize towards the ORIGINAL dense layer output (stored in targets),
            # not the compensated output. This allows RO to further refine the weights
            # after compensation to better match the original dense output.

        # 5b. Regional Optimization (RO) - SKIP if ro_iter is 0
        if args.ro_iter <= 0:
            print(f"  Skipping Regional Optimization (ro_iter={args.ro_iter})")
        else:
            print(f"  Running Regional Optimization for layer {i} ({args.ro_iter} iterations)...")
            
            # Identify parameters to optimize (only the ones we pruned/kept)
            params_to_opt = []
            for name in subset:
                params_to_opt.append(subset[name].weight)
                
            optimizer = Adam(params_to_opt, lr=args.ro_lr)
        
            if len(targets) > 0:
                sample_target = targets[0].to(dev)
                with torch.no_grad():
                    test_input = inps[0].unsqueeze(0).to(dev).float()
                    test_output = layer(test_input, attention_mask=attention_mask.to(dev), position_ids=position_ids.to(dev))[0]
                    if test_output.shape != sample_target.shape:
                        print(f"    ERROR: Output shape {test_output.shape} != target shape {sample_target.shape}")
                        print(f"    Skipping RO for this layer due to shape mismatch.")
                        # Continue to next layer, skip RO but still do weight check
                        pass
                    else:
                        print(f"    Output shape check OK: {test_output.shape}")
                        # DEBUG: Check initial loss BEFORE RO
                        initial_loss = nn.MSELoss()(test_output.float(), sample_target.float())
                        print(f"    DEBUG: Initial loss before RO: {initial_loss.item():.9f}")
                        print(f"    DEBUG: Target mean: {sample_target.mean().item():.6f}, std: {sample_target.std().item():.6f}")
                        print(f"    DEBUG: Output mean: {test_output.mean().item():.6f}, std: {test_output.std().item():.6f}")
                        diff = (test_output - sample_target).abs()
                        print(f"    DEBUG: Max diff: {diff.max().item():.9f}, Mean diff: {diff.mean().item():.9f}")
            
            # RO iterations
            for k in range(args.ro_iter):
                total_loss = 0
                valid_steps = 0
                
                # Use at most 32 samples for RO to reduce computation and improve stability
                ro_nsamples = min(32, args.nsamples)
                for j in range(ro_nsamples):
                    layer_input = inps[j].unsqueeze(0).to(dev).float()
                    target = targets[j].to(dev).float()
                    
                    # Check for NaNs in input
                    if torch.isnan(layer_input).any() or torch.isinf(layer_input).any():
                        continue
                    
                    # Check target for NaNs 
                    if torch.isnan(target).any() or torch.isinf(target).any():
                        continue
                    
                    optimizer.zero_grad()
                    with torch.amp.autocast(device_type='cuda', enabled=False):
                        out = layer(layer_input, attention_mask=attention_mask.to(dev), position_ids=position_ids.to(dev))[0]
                        out = out.float()
                    
                    # Check for NaNs in output
                    if torch.isnan(out).any() or torch.isinf(out).any():
                        continue
                    
                    if out.shape != target.shape:
                        continue

                    loss = nn.MSELoss()(out, target)
                    
                    if torch.isnan(loss) or torch.isinf(loss):
                        continue
                        
                    loss.backward()

                    grad_has_nan_inf = False
                    for param in params_to_opt:
                        if param.grad is not None:
                            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                                grad_has_nan_inf = True
                                break
                    
                    if grad_has_nan_inf:
                        optimizer.zero_grad()
                        continue
                    
                    # Clip gradients to prevent explosion
                    torch.nn.utils.clip_grad_norm_(params_to_opt, 0.5)
                    
                    optimizer.step()
                    
                    total_loss += loss.item()
                    valid_steps += 1
                    
                    # Enforce Sparsity (Zero out pruned weights)
                    with torch.no_grad():
                        for name in subset:
                            if args.unstr:
                                subset[name].weight.data *= masks[name].unsqueeze(0)
                
                if valid_steps > 0:
                    avg_loss = total_loss / valid_steps
                    print(f"    Iter {k+1}/{args.ro_iter}, Avg Loss: {avg_loss:.9f} (valid steps: {valid_steps})")
                else:
                    print(f"    Iter {k+1}/{args.ro_iter}: No valid steps.")
        
        # 6. Weight Compensation AFTER RO (if compensate is set but NOT compensate_first)
        if args.compensate and not getattr(args, 'compensate_first', False):
            print(f"  Compensating layer {i} (after RO)...")
            compensator = MagnitudeCompensator(model, ridge_lambda=args.ridge_lambda)
            grams = {}
            for e in entries:
                short_name = e['name'].split(f"layer_{i}_")[-1]
                grams[e['name']] = gram_recorders[short_name].gram
                gram_recorders[short_name].free()
            
            compensator.load_gram_stats(grams)
            compensator.load_compression_state(entries)
            compensator.compensate()
            
            del grams
            del entries
            del compensator
        
        # Free gram recorders if compensate_first was used
        if args.compensate and getattr(args, 'compensate_first', False):
            for short_name in gram_recorders:
                gram_recorders[short_name].free()
            del entries
            
        # 7. Cleanup & Update Inputs
        # Disable grad
        for name in subset:
            subset[name].weight.requires_grad = False
            subset[name].weight.grad = None
            
        # Update outputs for next layer
        for j in range(args.nsamples):
            with torch.no_grad():
                # Use FP32 for numerical stability
                layer_input = inps[j].unsqueeze(0).to(dev).float()
                with torch.amp.autocast(device_type='cuda', enabled=False):
                    layer_out = layer(layer_input, attention_mask=attention_mask.to(dev), position_ids=position_ids.to(dev))[0]
                    layer_out = layer_out.float()
                
                # Check for NaNs before propagating
                if torch.isnan(layer_out).any() or torch.isinf(layer_out).any():
                    print(f"    Warning: Layer {i} output contains NaN/Inf at sample {j}. Using input as fallback.")
                    layer_out = layer_input.squeeze(0)
                
                if inps.device.type == 'cpu':
                    outs[j] = layer_out.cpu()
                else:
                    outs[j] = layer_out
                    
        inps, outs = outs, inps
        torch.cuda.empty_cache()

    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()
