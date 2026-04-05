"""
optuna_search.py - Optuna Hyperparameter Search for LoRA Configuration.

Searches over LoRA rank, alpha, and optionally dropout to find the best
configuration that maximizes validation accuracy on CIFAR-100.

Usage:
    python optuna_search.py --n_trials 20 --epochs 5
    python optuna_search.py --n_trials 30 --epochs 10 --wandb_project my_project
"""

import argparse
import os
import gc
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import optuna
from optuna.exceptions import TrialPruned
import wandb
import json

from data import get_cifar100_loaders
from model import get_model
from utils import set_seed


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Optuna LoRA Hyperparameter Search")
    parser.add_argument('--n_trials', type=int, default=20, help='Number of Optuna trials')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Epochs per trial (use fewer for speed)')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--warmup_epochs', type=int, default=1, help='Warmup epochs')
    parser.add_argument('--data_root', type=str, default='./data', help='Data directory')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--results_dir', type=str, default='./results', help='Results directory')
    parser.add_argument('--wandb_project', type=str, default='dlops-assignment5-vit-lora',
                        help='WandB project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='WandB entity')
    parser.add_argument('--study_name', type=str, default='lora_hpo',
                        help='Optuna study name')
    return parser.parse_args()


def create_objective(args, train_loader, val_loader):
    """Create the Optuna objective function.

    Args:
        args: Parsed command-line arguments.
        train_loader: Training data loader.
        val_loader: Validation data loader.

    Returns:
        Objective function for Optuna.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def objective(trial):
        """Optuna objective function.

        Args:
            trial: Optuna trial object.

        Returns:
            Best validation accuracy achieved during training.
        """
        # Suggest LoRA hyperparameters
        rank = trial.suggest_categorical('rank', [2, 4, 8])
        alpha = trial.suggest_categorical('alpha', [2, 4, 8])
        dropout = trial.suggest_categorical('dropout', [0.05, 0.1, 0.15, 0.2])

        exp_name = f"optuna_trial{trial.number}_r{rank}_a{alpha}_d{dropout}"
        print(f"\n--- Trial {trial.number}: rank={rank}, alpha={alpha}, dropout={dropout} ---")

        # Initialize WandB for this trial
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=exp_name,
            group="optuna_search",
            config={
                "trial_number": trial.number,
                "rank": rank,
                "alpha": alpha,
                "dropout": dropout,
                "epochs": args.epochs,
                "lr": args.lr,
            },
            reinit=True,
        )

        # Create model with suggested hyperparameters
        model, trainable_params, total_params = get_model(
            use_lora=True, rank=rank, alpha=alpha, dropout=dropout, num_classes=100
        )
        model = model.to(device)

        # Loss, optimizer, scheduler
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay
        )

        warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=args.warmup_epochs)
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs)
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[args.warmup_epochs]
        )

        best_val_acc = 0.0

        for epoch in range(args.epochs):
            # Training
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            train_loss = running_loss / total
            train_acc = 100. * correct / total

            # Validation
            model.eval()
            val_loss_sum = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    val_loss_sum += loss.item() * images.size(0)
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()

            val_loss = val_loss_sum / val_total
            val_acc = 100. * val_correct / val_total

            scheduler.step()

            wandb.log({
                'epoch': epoch + 1,
                'train/loss': train_loss,
                'train/accuracy': train_acc,
                'val/loss': val_loss,
                'val/accuracy': val_acc,
            })

            print(f"  Epoch {epoch+1}/{args.epochs} | "
                  f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

            if val_acc > best_val_acc:
                best_val_acc = val_acc

            # Report to Optuna for pruning
            trial.report(val_acc, epoch)
            if trial.should_prune():
                wandb.finish()
                raise TrialPruned()

        wandb.run.summary["best_val_acc"] = best_val_acc
        wandb.finish()

        # Clean up GPU memory
        del model
        torch.cuda.empty_cache()
        gc.collect()

        return best_val_acc

    return objective


def main():
    """Main Optuna search function."""
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.results_dir, exist_ok=True)

    print("Loading CIFAR-100 data...")
    train_loader, val_loader, _, _ = get_cifar100_loaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed
    )

    # Create Optuna study
    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=2),
    )

    # Create objective function
    objective = create_objective(args, train_loader, val_loader)

    # Run optimization
    print(f"\nStarting Optuna search with {args.n_trials} trials...")
    start_time = time.time()
    study.optimize(objective, n_trials=args.n_trials)
    total_time = time.time() - start_time

    # Report results
    print(f"\n{'='*60}")
    print("OPTUNA SEARCH RESULTS")
    print(f"{'='*60}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Number of completed trials: {len(study.trials)}")
    print(f"\nBest trial:")
    print(f"  Value (Val Accuracy): {study.best_trial.value:.2f}%")
    print(f"  Params: {study.best_trial.params}")

    # Save results
    results = {
        "best_value": study.best_trial.value,
        "best_params": study.best_trial.params,
        "n_trials": len(study.trials),
        "total_time_seconds": total_time,
        "all_trials": [
            {
                "number": t.number,
                "value": t.value,
                "params": t.params,
                "state": str(t.state),
            }
            for t in study.trials
        ]
    }

    results_path = os.path.join(args.results_dir, "optuna_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved Optuna results to {results_path}")

    # Log final summary to WandB
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name="optuna_summary",
        job_type="hpo_summary",
        config=results["best_params"],
    )
    wandb.log({
        "best_val_accuracy": study.best_trial.value,
        "best_rank": study.best_trial.params["rank"],
        "best_alpha": study.best_trial.params["alpha"],
        "best_dropout": study.best_trial.params["dropout"],
        "n_completed_trials": len(study.trials),
    })
    wandb.finish()

    print(f"\nBest LoRA Configuration:")
    print(f"  Rank: {study.best_trial.params['rank']}")
    print(f"  Alpha: {study.best_trial.params['alpha']}")
    print(f"  Dropout: {study.best_trial.params['dropout']}")
    print(f"  Best Val Accuracy: {study.best_trial.value:.2f}%")


if __name__ == "__main__":
    main()
