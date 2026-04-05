"""
utils.py - Utility functions for training, evaluation, plotting, and logging.

Provides helpers for WandB logging, gradient monitoring, plotting training curves,
class-wise accuracy histograms, and result table generation.
"""

import os
import json
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for Docker
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import torch
import wandb


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def log_gradient_norms(model, step, use_lora=True):
    """Log gradient norms of LoRA weights (or all trainable) to WandB.

    Args:
        model: The model (PEFT-wrapped or plain).
        step: Current training step.
        use_lora: Whether model uses LoRA.
    """
    grad_norms = {}
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            if use_lora and ("lora_" in name):
                grad_norms[f"grad_norm/{name}"] = param.grad.norm().item()
            elif not use_lora:
                grad_norms[f"grad_norm/{name}"] = param.grad.norm().item()

    if grad_norms:
        wandb.log(grad_norms, step=step)


def plot_training_curves(history, save_dir, experiment_name):
    """Plot training/validation loss and accuracy curves.

    Args:
        history: Dict with keys 'train_loss', 'val_loss', 'train_acc', 'val_acc'.
        save_dir: Directory to save plots.
        experiment_name: Name for the plot title and filename.
    """
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss plot
    axes[0].plot(epochs, history['train_loss'], 'b-o', label='Train Loss', markersize=4)
    axes[0].plot(epochs, history['val_loss'], 'r-o', label='Val Loss', markersize=4)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'{experiment_name} - Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy plot
    axes[1].plot(epochs, history['train_acc'], 'b-o', label='Train Acc', markersize=4)
    axes[1].plot(epochs, history['val_acc'], 'r-o', label='Val Acc', markersize=4)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title(f'{experiment_name} - Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{experiment_name}_curves.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    # Log to WandB
    wandb.log({f"plots/{experiment_name}_curves": wandb.Image(save_path)})
    print(f"Saved training curves to {save_path}")


def plot_classwise_accuracy(class_accuracies, class_names, save_dir, experiment_name):
    """Plot class-wise test accuracy histogram.

    Args:
        class_accuracies: List/array of per-class accuracies.
        class_names: List of class name strings.
        save_dir: Directory to save plots.
        experiment_name: Name for the plot title and filename.
    """
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(20, 8))

    # Sort by accuracy for better visualization
    sorted_indices = np.argsort(class_accuracies)
    sorted_accs = np.array(class_accuracies)[sorted_indices]
    sorted_names = np.array(class_names)[sorted_indices]

    colors = plt.cm.RdYlGn(sorted_accs / 100.0)
    bars = ax.bar(range(len(sorted_accs)), sorted_accs, color=colors, width=0.8)

    ax.set_xlabel('Classes (sorted by accuracy)', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title(f'{experiment_name} - Class-wise Test Accuracy', fontsize=14)
    ax.set_xticks(range(len(sorted_names)))
    ax.set_xticklabels(sorted_names, rotation=90, fontsize=5)
    ax.axhline(y=np.mean(class_accuracies), color='blue', linestyle='--',
               label=f'Mean: {np.mean(class_accuracies):.1f}%')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{experiment_name}_classwise_acc.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    wandb.log({f"plots/{experiment_name}_classwise_accuracy": wandb.Image(save_path)})
    print(f"Saved class-wise accuracy histogram to {save_path}")


def plot_gradient_updates(grad_history, save_dir, experiment_name):
    """Plot gradient norm updates over training steps.

    Args:
        grad_history: Dict mapping layer_name -> list of gradient norms.
        save_dir: Directory to save plots.
        experiment_name: Name for the plot title and filename.
    """
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))

    for layer_name, norms in grad_history.items():
        # Simplify layer name for legend
        short_name = layer_name.split('.')[-3] + '.' + layer_name.split('.')[-2] + '.' + layer_name.split('.')[-1]
        ax.plot(norms, label=short_name, alpha=0.7, linewidth=0.5)

    ax.set_xlabel('Training Step')
    ax.set_ylabel('Gradient Norm')
    ax.set_title(f'{experiment_name} - LoRA Weight Gradient Updates')
    ax.legend(fontsize=6, ncol=2, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{experiment_name}_grad_updates.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    wandb.log({f"plots/{experiment_name}_gradient_updates": wandb.Image(save_path)})
    print(f"Saved gradient update plot to {save_path}")


def save_results_table(results_list, save_path):
    """Save experiment results as a formatted table (CSV + text).

    Args:
        results_list: List of dicts with experiment results.
        save_path: Path to save the results CSV.
    """
    df = pd.DataFrame(results_list)
    df.to_csv(save_path, index=False)

    # Also save a nicely formatted text table
    txt_path = save_path.replace('.csv', '.txt')
    with open(txt_path, 'w') as f:
        f.write(df.to_string(index=False))

    print(f"\nResults Table:")
    print(df.to_string(index=False))
    print(f"\nSaved to {save_path}")

    return df


def save_epoch_table(history, save_dir, experiment_name):
    """Save epoch-wise training results table.

    Args:
        history: Dict with training history.
        save_dir: Directory to save the table.
        experiment_name: Experiment identifier.
    """
    os.makedirs(save_dir, exist_ok=True)

    data = {
        'Epoch': list(range(1, len(history['train_loss']) + 1)),
        'Training Loss': [f"{x:.4f}" for x in history['train_loss']],
        'Validation Loss': [f"{x:.4f}" for x in history['val_loss']],
        'Training Accuracy': [f"{x:.2f}" for x in history['train_acc']],
        'Validation Accuracy': [f"{x:.2f}" for x in history['val_acc']],
    }

    df = pd.DataFrame(data)
    save_path = os.path.join(save_dir, f'{experiment_name}_epoch_table.csv')
    df.to_csv(save_path, index=False)

    # Log as WandB table
    wandb_table = wandb.Table(dataframe=df)
    wandb.log({f"tables/{experiment_name}_epochs": wandb_table})

    print(f"\n{experiment_name} - Epoch Results:")
    print(df.to_string(index=False))

    return df
