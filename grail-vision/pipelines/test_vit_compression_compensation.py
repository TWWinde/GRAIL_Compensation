import os
import sys
import glob
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import copy
from pathlib import Path
from compensation.prune_compensation import Vit_PruneCompensation
from compensation.folding_compensation import Vit_FoldingCompensation
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.tune_utils import repair_bn
from vit_pytorch import SimpleViT
from compression.fold import ViT_ModelFolding
from compression.mag_prune import ViT_MagnitudePruning
from compression.wanda_prune import ViT_WandaPruning
#from compression.wanda_compensation import ViT_WandaAwareFolding
from utils.eval_utils import test, count_parameters, get_outputs
from utils.tune_utils import retune_layernorm
from utils.logger import ExperimentLogger
import csv

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

def get_dataloaders():
    train_ds = datasets.CIFAR10("../data", train=True, download=True, transform=transform_train)
    val_ds = datasets.CIFAR10("../data", train=False, download=True, transform=transform_val)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=8, pin_memory=True)
    return train_loader, val_loader

# --------------------------------------------------------
# Model
# --------------------------------------------------------
def load_vit_model(checkpoint_path, device):
    """Load SimpleViT model and weights."""
    model = SimpleViT(
        image_size=32,
        patch_size=4,
        num_classes=10,
        dim=512,
        depth=6,
        heads=16,
        mlp_dim=512 * 2
    )
    state_dict = torch.load(checkpoint_path, map_location=device)
    if "last" in state_dict:
        model.load_state_dict(state_dict["last"])
    else:
        model.load_state_dict(state_dict)
    return model.to(device)

# --------------------------------------------------------
# Main
# --------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate ViT compression across ratios/checkpoints")
    parser.add_argument("--ckpt_dir", type=str, default="./checkpoints/vit")
    parser.add_argument("--method", type=str, default="fold",
                        choices=["fold", "mag-l1", "mag-l2", "wanda", "wandafold"])
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--save_data", action="store_true", help="Save experiment data for visualization")
    parser.add_argument("--log_dir", type=str, default="./experiment_logs/vit", help="Directory to save experiment logs")
    args = parser.parse_args()

    fix_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, val_loader = get_dataloaders()
    ckpt_paths = sorted(glob.glob(f"{args.ckpt_dir}/*.pth"))
    prune_ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    compressor_map = {
        "fold":   lambda m, r: ViT_ModelFolding(m, compression_ratio=r),
        "mag-l1": lambda m, r: ViT_MagnitudePruning(m, compression_ratio=r, p=1),
        "mag-l2": lambda m, r: ViT_MagnitudePruning(m, compression_ratio=r, p=2),
        "wanda": lambda m, r: ViT_WandaPruning(m, compression_ratio=r),
        # "wandafold": lambda m, r: ViT_WandaAwareFolding(m, compression_ratio=r),
    }

    compensator_map = {
        "fold":      lambda m: Vit_FoldingCompensation(m, ridge_lambda=1e-3),
        "mag-l1":    lambda m: Vit_PruneCompensation(m, ridge_lambda=1e-3),
        "mag-l2":    lambda m: Vit_PruneCompensation(m, ridge_lambda=1e-3),
        "wanda":    lambda m: Vit_PruneCompensation(m, ridge_lambda=1e-3),
       
    }

    # Initialize experiment data collection
    all_experiments_data = []
    
    for i, ckpt_path in enumerate(ckpt_paths):
        model_name = os.path.basename(ckpt_path)
        print(f"\n[MODEL] {i + 1}/{len(ckpt_paths)} {model_name} {args.method}")

        if args.save_data:
            experiment_name = f"{model_name}_{args.method}"
            log_loc = os.path.join(args.log_dir, args.method)
            os.makedirs(log_loc, exist_ok=True)
            logger = ExperimentLogger(experiment_name=experiment_name, log_dir=log_loc)
            logger.log_metadata(args.method, "cifar10")
        
        for ratio in prune_ratios:
            # Load model fresh
            model = load_vit_model(ckpt_path, device)
            orig_params = count_parameters(model)
            orig_acc = test(model, val_loader, device)

            # Baseline (only for ratio=0.0)
            if ratio == 0.0:
                if orig_acc < 10.0:  # sanity check
                    break
                log_line(ratio, "BASE", params=orig_params, acc=f"{orig_acc:.2f}")
                
                if args.save_data:
                    logger.log_ratio_data(ratio, orig_acc, orig_acc, orig_acc, orig_acc, orig_acc, , orig_acc, orig_acc, orig_params, orig_params)
                
                orig_outputs = get_outputs(model.eval(), val_loader, device)
                continue

            # Apply compression
            compressor = compressor_map[args.method](model, ratio)
            # Wanda needs calibration before apply()
           
            compressor.run_calibration(train_loader, device, num_batches=50)
            compre_model = compressor.apply().to(device)
            compre_model_1 = copy.deepcopy(compre_model)

            # get pruning state and gram stats
            compression_entries = compressor.get_compression_state()
            gram_stats = compressor.get_gram_stats()

            compressed_params = count_parameters(compre_model)
            compre_acc = test(compre_model, test_loader, device)
            log_line(ratio, "COMPRESSION", params=compressed_params, acc=f"{compre_acc:.2f}")

            # COMPRESSION, REAPIR
            retune_layernorm(compre_model_1, train_loader, device=DEVICE, lr=1e-4)
            compre_re_acc = test(compre_model_1, test_loader, device)
            log_line(ratio, "COMPRESSION, REAPIR", acc=f"{compre_re_acc:.2f}")

            # COMPRESSION, REPAIR, COMPENSATION
            compensator_ = compensator_map[args.method](compre_model_1) 
            compensator_.load_compression_state(compression_entries)
            compensator_.load_gram_stats(gram_stats)
            compre_compen_re_model = compensator_.apply()
            compre_re_compen_acc = test(compre_compen_re_model, test_loader, device)
            log_line(ratio, "COMPRESSION,  REPAIR, COMPENSATION", acc=f"{compre_re_compen_acc:.2f}")
            

            # COMPRESSION,  REPAIR, COMPENSATION, REPAIR
            retune_layernorm(compre_compen_re_model, train_loader, device=DEVICE, lr=1e-4)
            compre_re_compen_re_acc = test(compre_compen_re_model, test_loader, device)
            log_line(ratio, "COMPRESSION,  REPAIR, COMPENSATION, REPAIR", acc=f"{compre_re_compen_re_acc:.2f}")


            # COMPRESSION, COMPENSATION
            compensator = compensator_map[args.method](compre_model) 
            compensator.load_compression_state(compression_entries)
            compensator.load_gram_stats(gram_stats)
            compre_compen_model = compensator.apply()

            compre_compen_acc = test(compre_compen_model, test_loader, device)
            log_line(ratio, "COMPRESSION, COMPENSATION", acc=f"{compre_compen_acc:.2f}")

            # COMPRESSION, COMPENSATION, REPAIR
            retune_layernorm(compre_compen_model, train_loader, device=DEVICE, lr=1e-4)
            compre_compen_re_acc = test(compre_compen_model, test_loader, device)
            log_line(ratio, "COMPRESSION, COMPENSATION, REPAIR", acc=f"{compre_compen_re_acc:.2f}")

            # Save final performance data
            if args.save_data:
                layer_details = None
                compensated_params = count_parameters(compre_compen_model)
           
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
                    for x, y in train_loader:
                        x, y = x.to(device), y.to(device)
                        opt.zero_grad()
                        loss = loss_fn(model(x), y)
                        loss.backward()
                        opt.step()
                    acc = test(model, val_loader, device)
                    log_line(ratio, f"FINETUNE_EPOCH{epoch+1}", acc=f"{acc:.2f}")

 

            # Optional fine-tuning
            if args.epochs > 0:
                opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
                loss_fn = nn.CrossEntropyLoss()
                for epoch in range(args.epochs):
                    model.train()
                    for x, y in train_loader:
                        x, y = x.to(device), y.to(device)
                        opt.zero_grad()
                        loss = loss_fn(model(x), y)
                        loss.backward()
                        opt.step()
                    acc = test(model, val_loader, device)
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
