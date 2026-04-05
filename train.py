"""
train.py - Training script for ViT-S fine-tuning on CIFAR-100 with optional LoRA.

Supports:
- Fine-tuning classification head only (baseline, no LoRA)
- Fine-tuning with LoRA (various rank/alpha/dropout configs)
- WandB logging of losses, accuracies, gradient norms
- Checkpoint saving of best model

Usage:
    # Baseline (no LoRA):
    python train.py --no_lora --epochs 10

    # With LoRA:
    python train.py --rank 8 --alpha 8 --dropout 0.1 --epochs 10

    # With WandB project name:
    python train.py --rank 4 --alpha 4 --wandb_project my_project
"""

import argparse
import os
import sys
import time
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm
import wandb

from data import get_cifar100_loaders
from model import get_model, get_model_partial_frozen
from utils import (
    set_seed, log_gradient_norms, plot_training_curves,
    plot_classwise_accuracy, plot_gradient_updates,
    save_epoch_table
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="ViT-S CIFAR-100 Fine-tuning with LoRA")

    # Model/LoRA arguments
    parser.add_argument('--no_lora', action='store_true', help='Train without LoRA (baseline)')
    parser.add_argument('--rank', type=int, default=8, help='LoRA rank')
    parser.add_argument('--alpha', type=int, default=8, help='LoRA alpha')
    parser.add_argument('--dropout', type=float, default=0.1, help='LoRA dropout')
    parser.add_argument('--partial_frozen', action='store_true',
                        help='[Optional] Partial frozen + LoRA mode')
    parser.add_argument('--freeze_layers', type=int, default=6,
                        help='Number of layers to freeze in partial frozen mode')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--warmup_epochs', type=int, default=1, help='Warmup epochs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--val_split', type=float, default=0.1, help='Validation split ratio')

    # Data arguments
    parser.add_argument('--data_root', type=str, default='./data', help='Data directory')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')

    # Output arguments
    parser.add_argument('--save_dir', type=str, default='./weights', help='Weights save directory')
    parser.add_argument('--results_dir', type=str, default='./results', help='Results directory')

    # WandB arguments
    parser.add_argument('--wandb_project', type=str, default='dlops-assignment5-vit-lora',
                        help='WandB project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='WandB entity/username')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Experiment name (auto-generated if not provided)')

    # Config file
    parser.add_argument('--config', type=str, default=None, help='YAML config file path')

    return parser.parse_args()


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch,
                    use_lora, grad_history):
    """Train for one epoch.

    Args:
        model: The model to train.
        train_loader: Training data loader.
        criterion: Loss function.
        optimizer: Optimizer.
        device: torch device.
        epoch: Current epoch number.
        use_lora: Whether model uses LoRA.
        grad_history: Dict to track gradient norms.

    Returns:
        Tuple of (average_loss, accuracy_percentage).
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    global_step = epoch * len(train_loader)

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", leave=False)
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        # Track gradient norms for LoRA weights
        step = global_step + batch_idx
        if use_lora and (batch_idx % 10 == 0):  # Log every 10 steps
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None and "lora_" in name:
                    if name not in grad_history:
                        grad_history[name] = []
                    grad_history[name].append(param.grad.norm().item())

            log_gradient_norms(model, step, use_lora=True)

        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100. * correct / total:.2f}%'
        })

    avg_loss = running_loss / total
    accuracy = 100. * correct / total
    return avg_loss, accuracy


@torch.no_grad()
def validate(model, val_loader, criterion, device, epoch):
    """Validate the model.

    Args:
        model: The model to evaluate.
        val_loader: Validation data loader.
        criterion: Loss function.
        device: torch device.
        epoch: Current epoch number.

    Returns:
        Tuple of (average_loss, accuracy_percentage).
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = running_loss / total
    accuracy = 100. * correct / total
    return avg_loss, accuracy


def main():
    """Main training function."""
    args = parse_args()

    # Load config file if provided
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        # Config values are used as defaults, CLI args override
        for key, value in config.get('training', {}).items():
            if not hasattr(args, key) or getattr(args, key) is None:
                setattr(args, key, value)

    # Set seed
    set_seed(args.seed)

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Determine experiment name
    use_lora = not args.no_lora
    if args.experiment_name:
        exp_name = args.experiment_name
    elif args.partial_frozen:
        exp_name = f"partial_frozen_r{args.rank}_a{args.alpha}_d{args.dropout}"
    elif use_lora:
        exp_name = f"lora_r{args.rank}_a{args.alpha}_d{args.dropout}"
    else:
        exp_name = "baseline_no_lora"

    print(f"\n{'='*60}")
    print(f"Experiment: {exp_name}")
    print(f"{'='*60}\n")

    # Initialize WandB
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=exp_name,
        config={
            "use_lora": use_lora,
            "rank": args.rank if use_lora else None,
            "alpha": args.alpha if use_lora else None,
            "dropout": args.dropout if use_lora else None,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "warmup_epochs": args.warmup_epochs,
            "partial_frozen": args.partial_frozen,
            "freeze_layers": args.freeze_layers if args.partial_frozen else None,
        }
    )

    # Load data
    print("Loading CIFAR-100 data...")
    train_loader, val_loader, test_loader, class_names = get_cifar100_loaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        val_split=args.val_split,
        num_workers=args.num_workers,
        seed=args.seed
    )

    # Create model
    print("\nCreating model...")
    if args.partial_frozen:
        model, trainable_params, total_params = get_model_partial_frozen(
            rank=args.rank, alpha=args.alpha, dropout=args.dropout,
            num_classes=100, freeze_layers=args.freeze_layers
        )
    else:
        model, trainable_params, total_params = get_model(
            use_lora=use_lora,
            rank=args.rank, alpha=args.alpha, dropout=args.dropout,
            num_classes=100
        )

    model = model.to(device)
    wandb.config.update({
        "trainable_params": trainable_params,
        "total_params": total_params
    })

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Warmup + cosine annealing scheduler
    warmup_scheduler = LinearLR(
        optimizer, start_factor=0.1, total_iters=args.warmup_epochs
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=args.epochs - args.warmup_epochs
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[args.warmup_epochs]
    )

    # Training loop
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': []
    }
    grad_history = {}
    best_val_acc = 0.0
    best_epoch = 0

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    print(f"\nStarting training for {args.epochs} epochs...")
    start_time = time.time()

    for epoch in range(args.epochs):
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            epoch, use_lora, grad_history
        )

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device, epoch)

        # Update scheduler
        scheduler.step()

        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        # Log to WandB
        wandb.log({
            'epoch': epoch + 1,
            'train/loss': train_loss,
            'train/accuracy': train_acc,
            'val/loss': val_loss,
            'val/accuracy': val_acc,
            'lr': optimizer.param_groups[0]['lr']
        })

        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            save_path = os.path.join(args.save_dir, f"{exp_name}_best.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'config': vars(args),
            }, save_path)
            print(f"  -> Saved best model (val_acc: {val_acc:.2f}%)")

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.1f}s")
    print(f"Best validation accuracy: {best_val_acc:.2f}% (epoch {best_epoch})")

    # Plot training curves
    plot_training_curves(history, args.results_dir, exp_name)

    # Plot gradient updates for LoRA experiments
    if use_lora and grad_history:
        plot_gradient_updates(grad_history, args.results_dir, exp_name)

    # Save epoch table
    save_epoch_table(history, args.results_dir, exp_name)

    # Log summary
    wandb.run.summary["best_val_acc"] = best_val_acc
    wandb.run.summary["best_epoch"] = best_epoch
    wandb.run.summary["total_time_seconds"] = total_time
    wandb.run.summary["trainable_params"] = trainable_params

    wandb.finish()

    return best_val_acc, exp_name


if __name__ == "__main__":
    main()
