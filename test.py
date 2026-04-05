"""
test.py - Testing/Evaluation script for ViT-S on CIFAR-100.

Evaluates a trained model on the CIFAR-100 test set and reports:
- Overall test accuracy
- Class-wise test accuracy with histogram plot
- Detailed results table

Usage:
    # Test baseline model:
    python test.py --checkpoint weights/baseline_no_lora_best.pth --no_lora

    # Test LoRA model:
    python test.py --checkpoint weights/lora_r8_a8_d0.1_best.pth --rank 8 --alpha 8
"""

import argparse
import os
import json
import torch
import numpy as np
from tqdm import tqdm
import wandb

from data import get_cifar100_loaders
from model import get_model, get_model_partial_frozen
from utils import plot_classwise_accuracy, set_seed


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="ViT-S CIFAR-100 Testing")

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--no_lora', action='store_true',
                        help='Model was trained without LoRA')
    parser.add_argument('--rank', type=int, default=8, help='LoRA rank')
    parser.add_argument('--alpha', type=int, default=8, help='LoRA alpha')
    parser.add_argument('--dropout', type=float, default=0.1, help='LoRA dropout')
    parser.add_argument('--partial_frozen', action='store_true',
                        help='Model uses partial frozen mode')
    parser.add_argument('--freeze_layers', type=int, default=6,
                        help='Number of frozen layers')

    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--data_root', type=str, default='./data', help='Data directory')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--results_dir', type=str, default='./results',
                        help='Results directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    parser.add_argument('--wandb_project', type=str, default='dlops-assignment5-vit-lora',
                        help='WandB project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='WandB entity')

    return parser.parse_args()


@torch.no_grad()
def evaluate(model, test_loader, device, num_classes=100):
    """Evaluate model on test set.

    Args:
        model: Trained model.
        test_loader: Test data loader.
        device: torch device.
        num_classes: Number of classes.

    Returns:
        Tuple of (overall_accuracy, class_accuracies, class_correct, class_total).
    """
    model.eval()
    correct = 0
    total = 0

    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    pbar = tqdm(test_loader, desc="Testing")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)

        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        for i in range(labels.size(0)):
            label = labels[i].item()
            class_total[label] += 1
            if predicted[i].item() == label:
                class_correct[label] += 1

    overall_accuracy = 100. * correct / total

    class_accuracies = []
    for i in range(num_classes):
        if class_total[i] > 0:
            class_accuracies.append(100. * class_correct[i] / class_total[i])
        else:
            class_accuracies.append(0.0)

    return overall_accuracy, class_accuracies, class_correct, class_total


def main():
    """Main testing function."""
    args = parse_args()
    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    use_lora = not args.no_lora

    # Determine experiment name from checkpoint
    checkpoint_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
    exp_name = checkpoint_name.replace('_best', '')

    print(f"\n{'='*60}")
    print(f"Testing: {exp_name}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"{'='*60}\n")

    # Initialize WandB for test logging
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=f"test_{exp_name}",
        job_type="evaluation",
        config={
            "checkpoint": args.checkpoint,
            "use_lora": use_lora,
            "rank": args.rank if use_lora else None,
            "alpha": args.alpha if use_lora else None,
        }
    )

    # Load data
    print("Loading CIFAR-100 test data...")
    _, _, test_loader, class_names = get_cifar100_loaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed
    )

    # Create model with same architecture
    print("Creating model...")
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

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"Checkpoint val_acc: {checkpoint.get('val_acc', 'N/A'):.2f}%")

    # Evaluate
    print("\nRunning evaluation on test set...")
    overall_acc, class_accs, class_correct, class_total = evaluate(
        model, test_loader, device
    )

    print(f"\n{'='*60}")
    print(f"Overall Test Accuracy: {overall_acc:.2f}%")
    print(f"{'='*60}")

    # Plot class-wise accuracy
    os.makedirs(args.results_dir, exist_ok=True)
    plot_classwise_accuracy(class_accs, class_names, args.results_dir, exp_name)

    # Log to WandB
    wandb.log({
        "test/overall_accuracy": overall_acc,
        "test/trainable_params": trainable_params,
    })

    # Log class-wise accuracy table to WandB
    class_data = []
    for i, (name, acc) in enumerate(zip(class_names, class_accs)):
        class_data.append([name, acc, class_correct[i], class_total[i]])

    class_table = wandb.Table(
        columns=["Class Name", "Accuracy (%)", "Correct", "Total"],
        data=class_data
    )
    wandb.log({"test/classwise_accuracy_table": class_table})

    # Save results JSON
    results = {
        "experiment": exp_name,
        "use_lora": use_lora,
        "rank": args.rank if use_lora else None,
        "alpha": args.alpha if use_lora else None,
        "dropout": args.dropout if use_lora else None,
        "overall_test_accuracy": overall_acc,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "class_accuracies": {name: acc for name, acc in zip(class_names, class_accs)},
    }

    results_path = os.path.join(args.results_dir, f"{exp_name}_test_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved test results to {results_path}")

    wandb.finish()

    return overall_acc


if __name__ == "__main__":
    main()
