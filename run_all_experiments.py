"""
run_all_experiments.py - Run all LoRA experiments + baseline for Assignment 5.

Executes:
1. Baseline: Fine-tune classification head without LoRA
2. 9 LoRA experiments: Rank ∈ {2, 4, 8} × Alpha ∈ {2, 4, 8}, Dropout = 0.1
3. Generates summary results table

Usage:
    python run_all_experiments.py
    python run_all_experiments.py --epochs 10 --batch_size 64
"""

import argparse
import os
import sys
import json
import subprocess
import pandas as pd


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run all ViT-S LoRA experiments")
    parser.add_argument('--epochs', type=int, default=10, help='Epochs per experiment')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--data_root', type=str, default='./data', help='Data directory')
    parser.add_argument('--save_dir', type=str, default='./weights', help='Weights directory')
    parser.add_argument('--results_dir', type=str, default='./results', help='Results directory')
    parser.add_argument('--wandb_project', type=str, default='dlops-assignment5-vit-lora',
                        help='WandB project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='WandB entity')
    parser.add_argument('--skip_baseline', action='store_true', help='Skip baseline experiment')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')
    return parser.parse_args()


def run_experiment(cmd_args):
    """Run a single training experiment as a subprocess.

    Args:
        cmd_args: List of command-line arguments for train.py.

    Returns:
        Return code of the subprocess.
    """
    cmd = [sys.executable, "train.py"] + cmd_args
    print(f"\n{'='*80}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*80}\n")

    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    return result.returncode


def run_test(checkpoint, use_lora, rank=None, alpha=None, dropout=0.1,
             results_dir='./results', wandb_project='dlops-assignment5-vit-lora',
             wandb_entity=None, data_root='./data', num_workers=4):
    """Run testing for a trained model.

    Args:
        checkpoint: Path to model checkpoint.
        use_lora: Whether model uses LoRA.
        rank: LoRA rank (if applicable).
        alpha: LoRA alpha (if applicable).
        dropout: LoRA dropout.
        results_dir: Results output directory.
        wandb_project: WandB project name.
        wandb_entity: WandB entity.
        data_root: Data directory.
        num_workers: DataLoader workers.

    Returns:
        Return code of the subprocess.
    """
    cmd = [
        sys.executable, "test.py",
        "--checkpoint", checkpoint,
        "--results_dir", results_dir,
        "--wandb_project", wandb_project,
        "--data_root", data_root,
        "--num_workers", str(num_workers),
    ]

    if not use_lora:
        cmd.append("--no_lora")
    else:
        cmd.extend(["--rank", str(rank), "--alpha", str(alpha), "--dropout", str(dropout)])

    if wandb_entity:
        cmd.extend(["--wandb_entity", wandb_entity])

    print(f"\nTesting: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    return result.returncode


def main():
    """Run all experiments and generate summary."""
    args = parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    # Define LoRA hyperparameter combinations
    ranks = [2, 4, 8]
    alphas = [2, 4, 8]
    dropout = 0.1

    # Common training arguments
    common_args = [
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
        "--data_root", args.data_root,
        "--save_dir", args.save_dir,
        "--results_dir", args.results_dir,
        "--wandb_project", args.wandb_project,
        "--num_workers", str(args.num_workers),
    ]
    if args.wandb_entity:
        common_args.extend(["--wandb_entity", args.wandb_entity])

    experiments = []
    completed = []
    failed = []

    # ---- Experiment 0: Baseline (no LoRA) ----
    if not args.skip_baseline:
        print("\n" + "=" * 80)
        print("EXPERIMENT 0: Baseline (No LoRA) - Classification Head Only")
        print("=" * 80)

        exp_args = common_args + ["--no_lora"]
        ret = run_experiment(exp_args)
        exp_info = {
            "experiment": "Baseline (No LoRA)",
            "rank": "-",
            "alpha": "-",
            "dropout": "-",
            "use_lora": False,
        }

        if ret == 0:
            completed.append(exp_info)
            # Run test
            checkpoint = os.path.join(args.save_dir, "baseline_no_lora_best.pth")
            if os.path.exists(checkpoint):
                run_test(checkpoint, use_lora=False, results_dir=args.results_dir,
                         wandb_project=args.wandb_project, wandb_entity=args.wandb_entity,
                         data_root=args.data_root, num_workers=args.num_workers)
        else:
            failed.append(exp_info)

    # ---- LoRA Experiments ----
    exp_num = 1
    for rank in ranks:
        for alpha in alphas:
            print(f"\n{'='*80}")
            print(f"EXPERIMENT {exp_num}: LoRA Rank={rank}, Alpha={alpha}, Dropout={dropout}")
            print(f"{'='*80}")

            exp_args = common_args + [
                "--rank", str(rank),
                "--alpha", str(alpha),
                "--dropout", str(dropout),
            ]

            ret = run_experiment(exp_args)
            exp_info = {
                "experiment": f"LoRA Exp {exp_num}",
                "rank": rank,
                "alpha": alpha,
                "dropout": dropout,
                "use_lora": True,
            }

            if ret == 0:
                completed.append(exp_info)
                # Run test
                checkpoint = os.path.join(
                    args.save_dir, f"lora_r{rank}_a{alpha}_d{dropout}_best.pth"
                )
                if os.path.exists(checkpoint):
                    run_test(checkpoint, use_lora=True, rank=rank, alpha=alpha,
                             dropout=dropout, results_dir=args.results_dir,
                             wandb_project=args.wandb_project,
                             wandb_entity=args.wandb_entity,
                             data_root=args.data_root, num_workers=args.num_workers)
            else:
                failed.append(exp_info)

            exp_num += 1

    # ---- Generate Summary ----
    print(f"\n{'='*80}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*80}")

    # Collect all test results
    summary_results = []
    for exp in completed:
        if exp["use_lora"]:
            result_file = os.path.join(
                args.results_dir,
                f"lora_r{exp['rank']}_a{exp['alpha']}_d{exp['dropout']}_test_results.json"
            )
        else:
            result_file = os.path.join(args.results_dir, "baseline_no_lora_test_results.json")

        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                result = json.load(f)
            summary_results.append({
                "LoRA Layers": "With LoRA" if exp["use_lora"] else "Without LoRA",
                "Rank": exp["rank"],
                "Alpha": exp["alpha"],
                "Dropout": exp["dropout"],
                "Overall Test Accuracy": f"{result['overall_test_accuracy']:.2f}%",
                "Trainable Parameters": f"{result['trainable_params']:,}",
            })

    if summary_results:
        df = pd.DataFrame(summary_results)
        print("\nFinal Results Table:")
        print(df.to_string(index=False))

        # Save summary
        summary_path = os.path.join(args.results_dir, "summary_results.csv")
        df.to_csv(summary_path, index=False)
        print(f"\nSaved summary to {summary_path}")

    print(f"\nCompleted: {len(completed)}/{len(completed) + len(failed)} experiments")
    if failed:
        print(f"Failed experiments: {failed}")


if __name__ == "__main__":
    main()
