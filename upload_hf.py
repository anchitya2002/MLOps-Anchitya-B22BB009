"""
upload_hf.py - Upload Best Model Weights to HuggingFace Hub.

Uploads the best model checkpoint along with model card and configuration
to HuggingFace Model Hub.

Usage:
    python upload_hf.py --checkpoint weights/lora_r8_a8_d0.1_best.pth \
        --repo_name username/vit-s-cifar100-lora --hf_token YOUR_TOKEN
"""

import argparse
import os
import json
import torch
from huggingface_hub import HfApi, create_repo, upload_file, upload_folder
import tempfile
import shutil


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Upload model to HuggingFace Hub")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to best model checkpoint')
    parser.add_argument('--repo_name', type=str, required=True,
                        help='HuggingFace repo name (e.g., username/vit-s-cifar100-lora)')
    parser.add_argument('--hf_token', type=str, default=None,
                        help='HuggingFace API token (or set HF_TOKEN env var)')
    parser.add_argument('--results_dir', type=str, default='./results',
                        help='Results directory with test results')
    parser.add_argument('--private', action='store_true', help='Make repo private')
    return parser.parse_args()


def create_model_card(checkpoint_info, results_info=None):
    """Generate a model card markdown string.

    Args:
        checkpoint_info: Dict with model configuration info.
        results_info: Optional dict with test results.

    Returns:
        Model card as a markdown string.
    """
    config = checkpoint_info.get('config', {})
    use_lora = config.get('use_lora', True)
    rank = config.get('rank', 'N/A')
    alpha = config.get('alpha', 'N/A')
    dropout = config.get('dropout', 'N/A')
    val_acc = checkpoint_info.get('val_acc', 'N/A')

    card = f"""---
license: mit
tags:
  - image-classification
  - vision-transformer
  - cifar-100
  - lora
  - peft
  - fine-tuning
datasets:
  - cifar100
metrics:
  - accuracy
model-index:
  - name: ViT-S-CIFAR100-LoRA
    results:
      - task:
          type: image-classification
        dataset:
          name: CIFAR-100
          type: cifar100
        metrics:
          - name: Accuracy
            type: accuracy
            value: {val_acc if isinstance(val_acc, str) else f'{val_acc:.2f}'}
---

# ViT-S Fine-tuned on CIFAR-100 with LoRA

This model is a Vision Transformer Small (ViT-S/16) pretrained on ImageNet and
fine-tuned on CIFAR-100 using LoRA (Low-Rank Adaptation) via the PEFT library.

## Model Details

- **Base Model**: `vit_small_patch16_224` (timm)
- **Dataset**: CIFAR-100 (100 classes)
- **Fine-tuning Method**: {'LoRA (PEFT)' if use_lora else 'Classification Head Only'}
- **LoRA Configuration**:
  - Rank: {rank}
  - Alpha: {alpha}
  - Dropout: {dropout}
  - Target Modules: QKV attention weights

## Training Details

- **Epochs**: {config.get('epochs', 10)}
- **Batch Size**: {config.get('batch_size', 64)}
- **Learning Rate**: {config.get('lr', 1e-4)}
- **Optimizer**: AdamW
- **Scheduler**: Cosine Annealing with Warmup

## Results

- **Best Validation Accuracy**: {val_acc if isinstance(val_acc, str) else f'{val_acc:.2f}%'}

## Usage

```python
import timm
from peft import PeftModel, LoraConfig
import torch

# Load base model
base_model = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=100)

# Load fine-tuned weights
checkpoint = torch.load('best_model.pth', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

## Assignment Info

This model was trained as part of DLops Assignment 5.
"""
    return card


def main():
    """Upload model to HuggingFace Hub."""
    args = parse_args()

    # Get HF token
    hf_token = args.hf_token or os.environ.get('HF_TOKEN')
    if not hf_token:
        print("Error: HuggingFace token not provided. Use --hf_token or set HF_TOKEN env var.")
        return

    # Load checkpoint info
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')

    # Create temporary directory for upload
    upload_dir = os.path.join(args.results_dir, 'hf_upload')
    os.makedirs(upload_dir, exist_ok=True)

    # Copy checkpoint
    shutil.copy2(args.checkpoint, os.path.join(upload_dir, 'best_model.pth'))

    # Save model config
    config = checkpoint.get('config', {})
    with open(os.path.join(upload_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    # Create model card
    model_card = create_model_card(checkpoint)
    with open(os.path.join(upload_dir, 'README.md'), 'w') as f:
        f.write(model_card)

    # Create/get repo
    api = HfApi(token=hf_token)
    try:
        create_repo(
            repo_id=args.repo_name,
            token=hf_token,
            private=args.private,
            exist_ok=True,
        )
        print(f"Repository created/found: {args.repo_name}")
    except Exception as e:
        print(f"Warning creating repo: {e}")

    # Upload files
    print(f"Uploading to {args.repo_name}...")
    api.upload_folder(
        folder_path=upload_dir,
        repo_id=args.repo_name,
        token=hf_token,
    )

    print(f"\nModel uploaded successfully!")
    print(f"HuggingFace URL: https://huggingface.co/{args.repo_name}")

    # Clean up
    shutil.rmtree(upload_dir)


if __name__ == "__main__":
    main()
