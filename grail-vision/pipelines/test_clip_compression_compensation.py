import os
import sys
import glob
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.datasets import ImageNet
import copy
from pathlib import Path
from models.clip_vit import CLIPViT_B32
from compression.fold import CLIPViT_ModelFolding
from compression.mag_prune import CLIPViT_MagnitudePruning
from compression.wanda_prune import CLIPViT_WandaPruning
from compensation.prune_compensation import CLIPViT_PruneCompensation
from compensation.folding_compensation import CLIPViT_FoldingCompensation
#from compression.rand_fold import CLIPViT_RandomFolding
#from compression.rand_prune import CLIPViT_RandomPruning
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.tune_utils import repair_bn
# from compression.fold import ViT_ModelFolding # ToDo
from utils.eval_utils import test, count_parameters, get_outputs
from utils.tune_utils import retune_layernorm
from utils.logger import ExperimentLogger
import csv

DATA_ROOT = "./data"
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# --------------------------------------------------------
# Utils
# --------------------------------------------------------
def fix_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def log_line(ratio, event, **kwargs):
    """Unified concise logging"""
    parts = [f"RATIO={ratio:.1f}", f"EVENT={event}"]
    parts += [f"{k}={v}" for k, v in kwargs.items()]
    print(" ".join(parts))

# --------------------------------------------------------
# Data
# --------------------------------------------------------
norm = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(*norm),
])
transform_val = transforms.Compose([
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize(*norm),
])


# --------------------------------------------------------
# Model
# --------------------------------------------------------
def load_clip_vit_model(num_classes, checkpoint_path, device):
    """
    Load CLIP ViT-B/32 model and preprocessing transform.
    """
    clip_loader = CLIPViT_B32(
        checkpoint_path=checkpoint_path,
        num_classes=num_classes,
        device=device
    )

    return clip_loader.load()  # returns (model, preprocess)

# --------------------------------------------------------
# Main
# --------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate ViT compression across ratios/checkpoints")
    parser.add_argument("--ckpt_dir", type=str, default="./checkpoints/clipvit-b32-model-soups")
    parser.add_argument("--method", type=str, default="fold",
                        choices=["fold", "mag-l1", "mag-l2", "wanda", "wandafold"])
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--save_data", action="store_true", help="Save experiment data for visualization")
    parser.add_argument("--log_dir", type=str, default="./experiment_logs/vit", help="Directory to save experiment logs")
    parser.add_argument("--shard_id", type=int, default=0, help="Shard ID for distributed processing (0-indexed)")
    parser.add_argument("--num_shards", type=int, default=1, help="Total number of shards")
    args = parser.parse_args()

    fix_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")
    num_classes = 1000
    

    ckpt_paths = sorted(glob.glob(f"{args.ckpt_dir}/*.pt"))
    print(f"[INFO] Found {len(ckpt_paths)} checkpoints in {args.ckpt_dir}")

    # Sharding logic
    if args.num_shards > 1:
        chunk_size = int(np.ceil(len(ckpt_paths) / args.num_shards))
        start_idx = args.shard_id * chunk_size
        end_idx = min(start_idx + chunk_size, len(ckpt_paths))
        ckpt_paths = ckpt_paths[start_idx:end_idx]
        print(f"[INFO] Processing shard {args.shard_id + 1}/{args.num_shards}: {len(ckpt_paths)} checkpoints ({start_idx} to {end_idx})")
    compression_ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    compressor_map = {
        "fold":   lambda m, r: CLIPViT_ModelFolding(m, compression_ratio=r),
        "mag-l1": lambda m, r: CLIPViT_MagnitudePruning(m, compression_ratio=r),
        "mag-l2": lambda m, r: CLIPViT_MagnitudePruning(m, compression_ratio=r, p=2),
        "wanda": lambda m, r: CLIPViT_WandaPruning(m, compression_ratio=r),
        # "wandafold": lambda m, r: ViT_WandaAwareFolding(m, compression_ratio=r),
    }

    compensator_map = {
        "fold":      lambda m: CLIPViT_FoldingCompensation(m, ridge_lambda=1e-3),
        "mag-l1":    lambda m: CLIPViT_PruneCompensation(m, ridge_lambda=1e-3),
        "mag-l2":    lambda m: CLIPViT_PruneCompensation(m, ridge_lambda=1e-3),
        "wanda":    lambda m: CLIPViT_PruneCompensation(m, ridge_lambda=1e-3),
       
    }

    # Initialize experiment data collection
    all_experiments_data = []

    model, preprocess = load_clip_vit_model(num_classes, ckpt_paths[0], device)
    val_full = ImageNet(root=DATA_ROOT, split="val", transform=preprocess)

    n_total = len(val_full)
    n_val   = int(0.5 * n_total)
    n_test  = n_total - n_val

    generator = torch.Generator().manual_seed(42)
    val_dataset, test_dataset = random_split(val_full, [n_val, n_test], generator=generator)

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    for i, ckpt_path in enumerate(ckpt_paths):
        print(f"\n[INFO] Loading checkpoint: {ckpt_path}")
        model_name = os.path.basename(ckpt_path)
        print(f"\n[MODEL] {i + 1}/{len(ckpt_paths)} {model_name} {args.method}")
        base_model, preprocess = load_clip_vit_model(num_classes, ckpt_path, device)

        
        if args.save_data:
            experiment_name = f"{model_name}_{args.method}"
            log_loc = os.path.join(args.log_dir, args.method)
            os.makedirs(log_loc, exist_ok=True)
            logger = ExperimentLogger(experiment_name=experiment_name, log_dir=log_loc)
            logger.log_metadata(args.method, "cifar10")

        orig_params = count_parameters(base_model)
        orig_acc = test(base_model, test_loader, device)
        print(f" orig_params: {orig_params}")    
        for ratio in compression_ratios:
            # Load model fresh per ratio
            model = copy.deepcopy(base_model).to(device)
            
            
            # Baseline (only for ratio=0.0)
            if ratio == 0.0:
                if orig_acc < 10.0:  # sanity check
                    break
                log_line(ratio, "BASE", params=orig_params, acc=f"{orig_acc:.2f}")
                
                if args.save_data:
                    logger.log_ratio_data(ratio, orig_acc, orig_acc, orig_acc, orig_acc, orig_acc, orig_acc, orig_acc, orig_params, orig_params)
                
                orig_outputs = get_outputs(model.eval(), test_loader, device)
                continue

            # Apply compression
            compressor = compressor_map[args.method](model, ratio)
            # Wanda needs calibration before apply()
            if args.method == "wanda":
                forward_fn = getattr(model, "encode_image", None)
                compressor.run_calibration(val_loader, device, num_batches=50, forward_fn=forward_fn)
            else:
                compressor.run_calibration(val_loader, device)
            compre_model = compressor.apply().to(device)
            compre_model_1 = copy.deepcopy(compre_model)

            # get pruning state and gram stats
            compression_entries = compressor.get_compression_state()
            gram_stats = compressor.get_gram_stats()

            compressed_params = count_parameters(compre_model)
            print(f" compressed_params: {compressed_params}")
            retune_layernorm(compre_model, val_loader, device=DEVICE, lr=1e-4)
            compre_acc = test(compre_model, test_loader, device)
            log_line(ratio, "COMPRESSION", params=compressed_params, acc=f"{compre_acc:.2f}")

            # # COMPRESSION, REAPIR
            # repair_bn(compre_model_1, val_loader)
            # retune_layernorm(compre_model_1, val_loader, device=DEVICE, lr=1e-4)
            # compre_re_acc = test(compre_model_1, test_loader, device)
            # log_line(ratio, "COMPRESSION, REAPIR", acc=f"{compre_re_acc:.2f}")

            # # COMPRESSION, REPAIR, COMPENSATION
            # compensator_ = compensator_map[args.method](compre_model_1) 
            # compensator_.load_compression_state(compression_entries)
            # compensator_.load_gram_stats(gram_stats)
            # compre_compen_re_model = compensator_.apply()
            # compre_re_compen_acc = test(compre_compen_re_model, test_loader, device)
            # log_line(ratio, "COMPRESSION,  REPAIR, COMPENSATION", acc=f"{compre_re_compen_acc:.2f}")
            

            # # COMPRESSION,  REPAIR, COMPENSATION, REPAIR
            # retune_layernorm(compre_compen_re_model, val_loader, device=DEVICE, lr=1e-4)
            # compre_re_compen_re_acc = test(compre_compen_re_model, test_loader, device)
            # log_line(ratio, "COMPRESSION,  REPAIR, COMPENSATION, REPAIR", acc=f"{compre_re_compen_re_acc:.2f}")


            # COMPRESSION, COMPENSATION
            compensator = compensator_map[args.method](compre_model) 
            compensator.load_compression_state(compression_entries)
            compensator.load_gram_stats(gram_stats)
            compre_compen_model = compensator.apply()

            retune_layernorm(compre_compen_model, val_loader, device=DEVICE, lr=1e-4)
            compre_compen_acc = test(compre_compen_model, test_loader, device)
            log_line(ratio, "COMPRESSION, COMPENSATION", acc=f"{compre_compen_acc:.2f}")

            # # COMPRESSION, COMPENSATION, REPAIR
            # retune_layernorm(compre_compen_model, val_loader, device=DEVICE, lr=1e-4)
            # compre_compen_re_acc = test(compre_compen_model, test_loader, device)
            # log_line(ratio, "COMPRESSION, COMPENSATION, REPAIR", acc=f"{compre_compen_re_acc:.2f}")

            # Save final performance data
            if args.save_data:
                layer_details = None
                compensated_params = count_parameters(compre_compen_model)

                # Set None for ratios > 0 do not have these metrics
                compre_re_acc = None
                compre_re_compen_acc = None
                compre_re_compen_re_acc = None
                compre_compen_re_acc = None

                logger.log_ratio_data(ratio, orig_acc, compre_acc, compre_re_acc, compre_re_compen_acc, 
                compre_re_compen_re_acc, compre_compen_acc, compre_compen_re_acc, orig_params, compensated_params, layer_details)
                # Collect for summary
                experiment_summary = {
                    'model_name': model_name,
                    'compression_ratio': ratio,
                    'method': args.method,
                    'original_accuracy': orig_acc,
                    'compressed_accuracy': compre_acc,
                    'compressed_repaired_accuracy': compre_re_acc,
                    'compressed_repaired_compensated_accuracy': compre_re_compen_acc,
                    'compressed_repaired_compensated_repaired_accuracy': compre_re_compen_re_acc,
                    'compressed_compensated_accuracy': compre_compen_acc,
                    'compressed_compensated_repaired_accuracy': compre_compen_re_acc,
                    'original_params': orig_params,
                    'compressed_params': compensated_params
                }
                all_experiments_data.append(experiment_summary)

            # # Compute functional deviation
            # fd = torch.norm(orig_outputs - get_outputs(model.eval(), test_loader, device), dim=1).mean().item()
            # log_line(ratio, "FD", value=f"{fd:.4f}")

            # Optional fine-tune
            if args.epochs > 0:
                opt = torch.optim.Adam(model.parameters(), lr=args.lr)
                loss_fn = nn.CrossEntropyLoss()
                for epoch in range(args.epochs):
                    model.train()
                    for x, y in val_loader:
                        x, y = x.to(device), y.to(device)
                        opt.zero_grad()
                        loss = loss_fn(model(x), y)
                        loss.backward()
                        opt.step()
                    acc = test(model, test_loader, device)
                    log_line(ratio, f"FINETUNE_EPOCH{epoch+1}", acc=f"{acc:.2f}")


        if args.save_data:
            logger.save_json()
            logger.save_csv_summary()
            logger.save_layer_details_csv()
            print(f"[INFO] All data for {model_name} saved successfully!")

    # Save overall summary if data saving is enabled
    if args.save_data and all_experiments_data:
        summary_path = os.path.join(log_loc, "experiment_summary.csv")
        with open(summary_path, 'w', newline='') as f:
            fieldnames = all_experiments_data[0].keys()
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_experiments_data)
        print(f"\n[INFO] Overall experiment summary saved to: {summary_path}")

if __name__ == "__main__":
    main()
