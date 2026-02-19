import argparse
import os 
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from models.hf_llama.modeling_llama import LlamaForCausalLM

from importlib.metadata import version

from lib.prune import prune_wanda_sp, prune_flap, prune_magnitude_sp, prune_slimgpt, check_sparsity, prune_wanda_pp_sp
from lib.eval import eval_ppl

# Newton-based pruning (new method)
try:
    from newton_pruner.runner import run_newton_pruning
    NEWTON_AVAILABLE = True
except ImportError:
    NEWTON_AVAILABLE = False

"""python main.py \
    --model baffo32/decapoda-research-llama-7B-hf \
    --prune_method flap \
    --pruning_ratio 0.2 \
    --remove_heads -1 \
    --metrics WIFV \
    --structure AL-AM \
    --nsamples 1024 \
    --save_model "llm_weights/flap_p0.2_WIFV_ALAM_llama_7b/" \
    --eval \
"""
"""python main.py \
    --model meta-llama/Llama-2-7b-hf  \
    --prune_method flap \
    --pruning_ratio 0.2 \
    --remove_heads -1 \
    --metrics WIFV \
    --structure AL-AM \
    --nsamples 1024 \
    --save_model "llm_weights/flap_p0.2_WIFV_ALAM_llama_7b/" \
    --compensate \
    --eval \
"""

"""python main.py \
    --model TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T  \
    --prune_method flap \
    --pruning_ratio 0.5 \
    --remove_heads -1 \
    --metrics WIFV \
    --structure AL-AM \
    --nsamples 1024 \
    --save_model "llm_weights/flap_p0.2_WIFV_ALAM_llama_7b/" \
    --eval \
"""

"""python main.py \
    --model huggyllama/llama-7b  \
    --prune_method flap \
    --pruning_ratio 0.5 \
    --remove_heads -1 \
    --metrics WIFV \
    --structure AL-AM \
    --nsamples 1024 \
    --save_model "llm_weights/flap_p0.2_WIFV_ALAM_llama_7b/" \
    --dataset wikitext2 \
    --zero_shot \
    --eval \


python main.py \
    --model meta-llama/Llama-2-7b-hf \
    --prune_method slimgpt \
    --pruning_ratio 0.5 \
    --dataset wikitext2 \
    --compensate \
    --eval

    
""" 

print('torch', version('torch'))
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count())

def get_llm(model, cache_dir="llm_weights"):
    # model = AutoModelForCausalLM.from_pretrained(
    #     model, 
    #     torch_dtype=torch.float16, 
    #     cache_dir=cache_dir, 
    #     low_cpu_mem_usage=True, 
    #     device_map="auto"
    # )
    model = LlamaForCausalLM.from_pretrained(
        model, 
        torch_dtype=torch.float16, 
        cache_dir=cache_dir, 
        low_cpu_mem_usage=True, 
        # device_map="auto"
        use_safetensors=True
    )
    
    for i in range(32):
        model.model.layers[i].self_attn.o_proj.bias = torch.nn.Parameter(torch.zeros_like(model.model.layers[i].self_attn.o_proj.bias, device='cpu'))  # 或 'cuda'
        model.model.layers[i].mlp.down_proj.bias = torch.nn.Parameter(torch.zeros_like(model.model.layers[i].mlp.down_proj.bias, device='cpu'))  # 或 'cuda'
        torch.nn.init.zeros_(model.model.layers[i].self_attn.o_proj.bias)
        torch.nn.init.zeros_(model.model.layers[i].mlp.down_proj.bias)
        
    model.seqlen = 2048
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='LLaMA model')    # Huggingface model name
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')
    parser.add_argument('--dataset', type=str, default="wikitext2", choices=['wikitext2', 'ptb', 'c4'], help='Dataset for calibration and compensation.')
    parser.add_argument('--eval_dataset', type=str, default=None, choices=['wikitext2', 'ptb', 'c4'], help='Dataset for evaluation. If not specified, uses --dataset.')
    parser.add_argument('--pruning_ratio', type=float, default=0, help='Pruning ratio.')
    parser.add_argument('--remove_heads', type=int, default=8, help='Remove num_heads')
    parser.add_argument("--metrics", type=str, default="WIFV", choices=["IFV", "WIFV", "WIFN", 'N/A'])
    parser.add_argument("--structure", type=str, default="AL-AM", choices=["UL-UM", "UL-MM", "AL-MM", "AL-AM", 'N/A'])
    parser.add_argument("--prune_method", type=str, default="flap", choices=["flap", "wanda_sp", "mag_sp", "slimgpt", "wanda_pp_sp", "newton_sp"])
    parser.add_argument("--cache_dir", default="llm_weights", type=str)
    parser.add_argument('--unstr', action="store_true")
    parser.add_argument('--compensate', action="store_true", help="Apply weight compensation")
    parser.add_argument('--eval', action="store_true", help="Evaluate")
    parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')
    parser.add_argument('--ridge_lambda', type=float, default=1e-3, help='Ridge regression lambda for compensation.')
    parser.add_argument('--zero_shot', action="store_true", help="Run zero-shot evaluation on common tasks.")
    parser.add_argument('--eval_batch_size', type=int, default=16, help='Batch size for zero-shot evaluation.')
    parser.add_argument('--no_bias_compensation', action="store_true", help="Disable bias compensation in FLAP.")
    parser.add_argument('--comp_nsamples', type=int, default=128, help='Number of samples for compensation Gram matrix. Default: 128.')
    
    # Wanda++ arguments
    parser.add_argument('--alpha', type=float, default=100, help='Alpha parameter for Wanda++ RGS.')
    parser.add_argument('--ro_iter', type=int, default=5, help='Number of iterations for Regional Optimization in Wanda++.')
    parser.add_argument('--ro_lr', type=float, default=3e-7, help='Learning rate for Regional Optimization in Wanda++.')
    parser.add_argument('--compensate_first', action='store_true', help='Run compensation BEFORE Regional Optimization in Wanda++.')

    # Newton pruning arguments
    parser.add_argument('--newton_iterations', type=int, default=50, help='Number of Newton iterations for mask optimization.')
    parser.add_argument('--lambda_penalty', type=float, default=1.0, help='Lagrange multiplier for budget constraint in Newton pruning.')
    parser.add_argument('--damping', type=float, default=1e-4, help='Hessian damping factor for Newton pruning.')
    parser.add_argument('--balance_factor', type=float, default=1.0, help='Attention vs MLP balance factor for Newton pruning.')
    parser.add_argument('--global_threshold', action='store_true', help='Use global threshold instead of per-layer for Newton pruning.')

    args = parser.parse_args()
    print(args)
    
    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    # Build the model and tokenizer
    print(f"loading llm model {args.model}")
    model = get_llm(args.model, args.cache_dir)
    device = torch.device("cuda:0")
    model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    if "30b" in args.model or "65b" in args.model: # for 30b and 65b we use device_map to load onto multiple A6000 GPUs, thus the processing here.
        device = model.hf_device_map["lm_head"]
    print("use device ", device)

    # Prune the model
    print("pruning starts")
    if args.prune_method == "flap":
        if args.metrics == 'N/A':
            raise ValueError("For FLAP pruning, the metrics parameter must be chosen from ['IFV', 'WIFV', 'WIFN']. 'N/A' is not a valid choice.")  
        if args.structure == 'N/A':
            raise ValueError("For FLAP pruning, the compressed model structure parameter must be chosen from ['UL-UM', 'UL-MM', 'AL-MM', 'AL-AM']. 'N/A' is not a valid choice.")  
        prune_flap(args, model, tokenizer, device)
    elif args.prune_method == "wanda_sp":
        prune_wanda_sp(args, model, tokenizer, device)
    elif args.prune_method == "mag_sp":
        prune_magnitude_sp(args, model, tokenizer, device)
    elif args.prune_method == "slimgpt":
        prune_slimgpt(args, model, tokenizer, device)
    elif args.prune_method == "wanda_pp_sp":
        prune_wanda_pp_sp(args, model, tokenizer, device)
    elif args.prune_method == "newton_sp":
        if not NEWTON_AVAILABLE:
            raise ImportError("Newton pruning module not available. Check newton_pruner package.")
        run_newton_pruning(model, tokenizer, device, args)

    # Check the sparsity of the model
    print("*"*30)
    sparsity_ratio = check_sparsity(model)
    print(f"sparsity sanity check {sparsity_ratio:.4f}")
    print(f"model parameter {sum(p.numel() for p in model.parameters()) / 1000 ** 3:.2f}B")
    print("*"*30)
    # Evaluate the model
    if args.eval:
        eval_dataset = args.eval_dataset if args.eval_dataset else args.dataset
        ppl = eval_ppl(model, tokenizer, device, eval_dataset)    
        print(f"ppl on {eval_dataset} {ppl}")

    # Zero-shot Evaluation
        # Zero-shot Evaluation
    if args.zero_shot:
        print("*" * 30)
        print("Running Zero-shot Evaluation...")

        try:
            import lm_eval
            from lm_eval import evaluator
            from lm_eval.models.huggingface import HFLM
            import re

            # ----- NORMALIZATION HELPER -----
            def normalize_output(text, task):
                t = text.strip().lower()

                # BoolQ: expect "yes" or "no"
                if task == "boolq":
                    if "yes" in t:
                        return "yes"
                    if "no" in t:
                        return "no"
                    if "true" in t:
                        return "yes"
                    if "false" in t:
                        return "no"
                    return "yes"

                # Winogrande / PIQA: expect "1" or "2"
                if task in ["winogrande", "piqa"]:
                    m = re.search(r"\b([12])\b", t)
                    if m:
                        return m.group(1)
                    if "a" in t:
                        return "1"
                    if "b" in t:
                        return "2"
                    return "1"

                # Multiple-choice: ARC / HellaSwag expect A/B/C/D
                if task in ["arc_easy", "arc_challenge", "hellaswag"]:
                    m = re.search(r"\b([abcd])\b", t)
                    if m:
                        return m.group(1).upper()
                    m = re.search(r"[A-D]", text)
                    if m:
                        return m.group(0)
                    return "A"

                return text

            # ----- CUSTOM NORMALIZED LLM WRAPPER -----
            class NormalizedHFLM(HFLM):
                def generate_until(self, requests):
                    # call original generate()
                    outputs = super().generate_until(requests)

                    # extract task_name for each sample
                    task_names = [req.args[2]["task_name"] for req in requests]

                    # normalize each output
                    normalized = []
                    for out, task in zip(outputs, task_names):
                        norm = normalize_output(out, task)
                        normalized.append(norm)

                    return normalized

            # ----- RUN LM-EVAL -----
            task_names = ["piqa", "hellaswag", "winogrande", "arc_easy", "arc_challenge", "boolq"]
            print(f"Evaluating on tasks: {task_names}")

            hflm = NormalizedHFLM(
                pretrained=model,
                tokenizer=tokenizer,
                batch_size=args.eval_batch_size
            )

            results = evaluator.simple_evaluate(
                model=hflm,
                tasks=task_names,
                num_fewshot=0,
                batch_size=args.eval_batch_size,
                device=device
            )

            # ----- PRINT RESULTS -----
            print("\nZero-shot Results:")
            for task, res in results["results"].items():
                acc = res.get("acc,none", res.get("acc", 0.0))
                acc_norm = res.get("acc_norm,none", res.get("acc_norm", 0.0))
                print(f"{task}: Acc={acc:.4f}, Acc_Norm={acc_norm:.4f}")

        except ImportError:
            print("Error: lm_eval not installed. Please install it to run zero-shot evaluation.")
        except Exception as e:
            print(f"Error during zero-shot evaluation: {e}")

        print("*" * 30)
        
    # Save the model
    if args.save_model:
        if not os.path.exists(args.save_model):
            os.makedirs(args.save_model)
        # torch.save(model, f'{args.save_model}/pruned_model.pt')
        model.save_pretrained(args.save_model)
        tokenizer.save_pretrained(args.save_model)
    

if __name__ == '__main__':
    main()