import os
import argparse
import random
import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v2, mobilenet_v3_small, resnet18, alexnet, vgg11_bn
import wandb

# ---------------------- Argparse ----------------------
parser = argparse.ArgumentParser(description="Train a CIFAR-10 model with configurable options.")
parser.add_argument("--opt", type=str, required=True, choices=["adam", "sgd", "rmsprop"])
parser.add_argument("--arch", type=str, required=True, choices=["mobilenet_v2", "mobilenet_v3_small", "resnet18", "alexnet", "vgg11_bn"])
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--lr", type=float, required=True)
parser.add_argument("--momentum", type=float, required=True)
parser.add_argument("--wd", type=float, required=True)
parser.add_argument("--batch_size", type=int, required=True)
parser.add_argument("--epochs", type=int, required=True)
parser.add_argument("--rand_aug", action="store_true")
parser.add_argument("--l1_lambda", type=float, default=0.0)
parser.add_argument("--l2_lambda", type=float, default=0.0)
parser.add_argument("--sam", action="store_true")
parser.add_argument("--sam_rho", type=float, default=0.05)
parser.add_argument("--lr_schedule", action="store_true")
args = parser.parse_args()

print("CONFIG:", vars(args))

# ---------------------- Fix Seed ----------------------
def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
fix_seed(args.seed)

# ---------------------- Model Head Replacement ----------------------
def replace_head(model, arch, num_classes):
    if arch == 'mobilenet_v2':
        model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    elif arch == 'mobilenet_v3_small':
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    elif arch == 'resnet18':
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif arch == 'alexnet':
        model.features[0] = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        model.features[2] = nn.Identity()
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    elif arch == 'vgg11_bn':
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    return model

# ---------------------- SAM Optimizer ----------------------
class SAM(torch.optim.Optimizer):
    def __init__(self, base_optimizer_cls, model, rho=0.05, **kwargs):
        self.model = model
        self.rho = rho
        self.base_optimizer = base_optimizer_cls(model.parameters(), **kwargs)
        self.param_backup = {}

    def first_step(self):
        grad_norm = torch.norm(torch.stack([p.grad.norm() for p in self.model.parameters() if p.grad is not None]), p=2)
        for p in self.model.parameters():
            if p.grad is None: continue
            e_w = p.grad * self.rho / (grad_norm + 1e-12)
            self.param_backup[p] = p.data.clone()
            p.data.add_(e_w)

    def second_step(self):
        for p in self.model.parameters():
            if p in self.param_backup:
                p.data = self.param_backup[p]
        self.base_optimizer.step()
        self.param_backup.clear()

    def step(self):
        raise NotImplementedError("Use first_step and second_step")

    def zero_grad(self):
        self.base_optimizer.zero_grad()

# ---------------------- Data ----------------------
def get_transforms(rand_aug):
    augments = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.RandomRotation(15)]
    if rand_aug:
        augments.append(transforms.RandAugment())
    transform_train = transforms.Compose(augments + [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    transform_test = transforms.Compose([
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    return transform_train, transform_test

def get_dataloaders(batch_size, rand_aug):
    t_train, t_test = get_transforms(rand_aug)
    d_train = datasets.CIFAR10(root="data", train=True, download=True, transform=t_train)
    d_test = datasets.CIFAR10(root="data", train=False, download=True, transform=t_test)
    return DataLoader(d_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True), \
           DataLoader(d_test, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

# ---------------------- Train + Eval ----------------------
def train_and_evaluate(model, optimizer, scheduler, criterion, device, train_loader, test_loader, num_epochs, l1_lambda):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            if l1_lambda > 0:
                loss += l1_lambda * sum(p.abs().sum() for p in model.parameters())
            loss.backward()
            if isinstance(optimizer, SAM):
                optimizer.first_step()
                model.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.second_step()
            else:
                optimizer.step()
            train_loss += loss.item()
            correct += (outputs.argmax(1) == targets).sum().item()
            total += targets.size(0)

        acc = correct / total
        if scheduler: scheduler.step()

        # Eval
        model.eval()
        test_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
                correct += (outputs.argmax(1) == targets).sum().item()
                total += targets.size(0)
        test_acc = correct / total

        print(f"Epoch {epoch + 1}, Train Loss: {train_loss / len(train_loader):.4f}, Train Acc: {acc*100:.2f}%, Test Loss: {test_loss / len(test_loader):.4f}, Test Acc: {test_acc*100:.2f}%")
        wandb.log({"train_loss": train_loss / len(train_loader), "train_acc": acc, "test_loss": test_loss / len(test_loader), "test_acc": test_acc})

# ---------------------- Main ----------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model = {
    "mobilenet_v2": mobilenet_v2,
    "mobilenet_v3_small": mobilenet_v3_small,
    "resnet18": resnet18,
    "alexnet": alexnet,
    "vgg11_bn": vgg11_bn,
}[args.arch](weights=None)
model = replace_head(model, args.arch, num_classes=10)

base_cls = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD, "rmsprop": torch.optim.RMSprop}[args.opt]
opt_kwargs = {"lr": args.lr, "weight_decay": args.wd}
if args.opt in ["sgd", "rmsprop"]:
    opt_kwargs["momentum"] = args.momentum

optimizer = SAM(base_cls, model, rho=args.sam_rho, **opt_kwargs) if args.sam else base_cls(model.parameters(), **opt_kwargs)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer.base_optimizer if args.sam else optimizer, T_max=args.epochs, eta_min=0) if args.lr_schedule else None
criterion = nn.CrossEntropyLoss()

wandb.init(project="FuncOpt", name=f"{args.arch}_{args.opt}_seed{args.seed}", config=vars(args))
train_loader, test_loader = get_dataloaders(args.batch_size, args.rand_aug)
train_and_evaluate(model, optimizer, scheduler, criterion, device, train_loader, test_loader, args.epochs, args.l1_lambda)

# Save final model
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
os.makedirs(f"checkpoints/{args.arch}/{args.opt}", exist_ok=True)
torch.save(model.state_dict(),
           f"checkpoints/{args.arch}/{args.opt}/{timestamp}_dataset=cifar10"
           f"_arch={args.arch}_opt={args.opt}_seed={args.seed}_lr={args.lr}_batch_size={args.batch_size}"
           f"_momentum={args.momentum}_wd={args.wd}_epochs={args.epochs}"
           f"_l1={args.l1_lambda}_l2={args.l2_lambda}_sam={args.sam}_sam_rho={args.sam_rho}"
           f"_rand_aug={args.rand_aug}_lr_schedule={args.lr_schedule}.pth")

wandb.finish()
