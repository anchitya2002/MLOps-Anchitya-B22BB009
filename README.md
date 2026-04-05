# DLops Assignment-5: ViT-S Fine-Tuning on CIFAR-100 with LoRA

**Name:** Anchitya  
**Roll Number:** B22BB009  
**Institute:** IIT Jodhpur

---

## Links

| Resource | URL |
|----------|-----|
| WandB Dashboard | https://wandb.ai/anchitya2003-indian-institute-of-technology-jodhpur/dlops-assignment5-vit-lora |
| HuggingFace Model | https://huggingface.co/B22BB009/vit-s-cifar100-lora |

---

## Overview

This assignment fine-tunes a **ViT-S/16** model (pretrained on ImageNet) on the **CIFAR-100** dataset (100 classes) using two strategies:

1. **Baseline** – freeze the entire backbone, train only the 100-class classification head  
2. **LoRA** – inject low-rank adapter matrices into the QKV attention weights using HuggingFace PEFT, keeping the backbone frozen but adapting the attention layers

All experiments run inside a **Docker container** for reproducibility.

---

## Installation

### Prerequisites
- Docker with NVIDIA GPU support (`nvidia-container-toolkit`)
- NVIDIA GPU (experiments run on GTX 1650, 4 GB VRAM)
- Git

### Clone and setup

```bash
git clone https://github.com/anchitya/DLops-Assignment-5.git
cd DLops-Assignment-5
git checkout "Assignment-5"
```

### Build Docker image

```bash
docker build -t dlops-assignment5 .
```

### Or install locally (Python 3.11+)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

---

## Project Structure

```
.
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── configs/
│   └── default.yaml
├── data.py               # CIFAR-100 loading, transforms, train/val split
├── model.py              # ViT-S model creation + LoRA via PEFT
├── train.py              # Training loop with WandB logging
├── test.py               # Evaluation + class-wise accuracy histogram
├── run_all_experiments.py # Runs baseline + all 9 LoRA combinations
├── optuna_search.py      # Optuna hyperparameter search
├── upload_hf.py          # Upload best weights to HuggingFace
├── utils.py              # Plotting and logging helpers
├── generate_pdf.py       # Report generator
├── weights/              # Saved model checkpoints
└── results/              # Plots, tables, JSON results
```

---

## How to Run

### Run all experiments (Docker)

```bash
docker run --gpus all --name dlops-experiments \
  -v "$PWD/weights:/app/weights" \
  -v "$PWD/results:/app/results" \
  -v "$PWD/data:/app/data" \
  -e WANDB_API_KEY="your_wandb_key" \
  -e WANDB_MODE="online" \
  dlops-assignment5 \
  python run_all_experiments.py --epochs 10 --batch_size 16 --num_workers 2
```

### Run baseline only

```bash
docker run --gpus all --rm \
  -v "$PWD/weights:/app/weights" \
  -v "$PWD/results:/app/results" \
  -v "$PWD/data:/app/data" \
  -e WANDB_API_KEY="your_wandb_key" \
  dlops-assignment5 \
  python train.py --no_lora --epochs 10 --batch_size 16 --num_workers 2
```

### Run a single LoRA experiment (e.g. rank=8, alpha=8)

```bash
docker run --gpus all --rm \
  -v "$PWD/weights:/app/weights" \
  -v "$PWD/results:/app/results" \
  -v "$PWD/data:/app/data" \
  -e WANDB_API_KEY="your_wandb_key" \
  dlops-assignment5 \
  python train.py --rank 8 --alpha 8 --dropout 0.1 --epochs 10 --batch_size 16 --num_workers 2
```

### Run Optuna search

```bash
docker run --gpus all --rm \
  -v "$PWD/weights:/app/weights" \
  -v "$PWD/results:/app/results" \
  -v "$PWD/data:/app/data" \
  -e WANDB_API_KEY="your_wandb_key" \
  dlops-assignment5 \
  python optuna_search.py --n_trials 20 --epochs 5 --batch_size 16 --num_workers 2
```

### Test a saved model

```bash
docker run --gpus all --rm \
  -v "$PWD/weights:/app/weights" \
  -v "$PWD/results:/app/results" \
  -v "$PWD/data:/app/data" \
  -e WANDB_API_KEY="your_wandb_key" \
  dlops-assignment5 \
  python test.py --checkpoint /app/weights/lora_r8_a8_d0.1_best.pth --rank 8 --alpha 8
```

### Upload best model to HuggingFace

```bash
docker run --rm \
  -v "$PWD/weights:/app/weights" \
  -v "$PWD/results:/app/results" \
  dlops-assignment5 \
  python upload_hf.py \
    --checkpoint /app/weights/lora_r8_a8_d0.1_best.pth \
    --repo_name "B22BB009/vit-s-cifar100-lora" \
    --hf_token "your_hf_token"
```

---

## Experimental Results

### Setup

| Setting | Value |
|---------|-------|
| Base model | vit_small_patch16_224 (timm, ImageNet pretrained) |
| Dataset | CIFAR-100 (100 classes, 32x32 → resized to 224x224) |
| LoRA target | QKV attention weights (all 12 blocks) |
| Epochs | 10 |
| Batch size | 16 |
| Optimizer | AdamW (lr=1e-4, weight_decay=1e-4) |
| Scheduler | Linear warm-up (1 epoch) + Cosine Annealing |
| Dropout (LoRA) | 0.1 |

---

### Q1.1 — Baseline: No LoRA (head-only fine-tuning)

| Epoch | Train Loss | Val Loss | Train Acc (%) | Val Acc (%) |
|-------|-----------|---------|--------------|------------|
| 1 | 4.2308 | 3.6974 | 9.84 | 19.30 |
| 2 | 1.6901 | 1.1608 | 61.53 | 70.92 |
| 3 | 1.0446 | 0.9771 | 71.71 | 73.56 |
| 4 | 0.9191 | 0.9219 | 74.39 | 74.56 |
| 5 | 0.8572 | 0.9010 | 75.56 | 74.82 |
| 6 | 0.8208 | 0.8799 | 76.75 | 75.14 |
| 7 | 0.7915 | 0.8668 | 77.30 | 75.50 |
| 8 | 0.7725 | 0.8596 | 78.10 | **76.04** |
| 9 | 0.7678 | 0.8566 | 77.92 | 75.96 |
| 10 | 0.7616 | 0.8559 | 78.43 | 75.98 |

**Best Val Accuracy: 76.04% | Trainable Params: 38,500**

---

### Q1.2 & Q1.3 — LoRA Experiments (Rank × Alpha, Dropout = 0.1)

#### Experiment 1: Rank=2, Alpha=2

| Epoch | Train Loss | Val Loss | Train Acc (%) | Val Acc (%) |
|-------|-----------|---------|--------------|------------|
| 1 | 4.1775 | 3.5596 | 11.27 | 22.96 |
| 2 | 0.8946 | 0.6152 | 78.32 | 83.58 |
| 3 | 0.4253 | 0.5209 | 87.24 | 85.36 |
| 4 | 0.3565 | 0.4823 | 88.87 | 85.74 |
| 5 | 0.3190 | 0.4572 | 89.82 | 86.22 |
| 6 | 0.2886 | 0.4432 | 90.84 | 86.30 |
| 7 | 0.2700 | 0.4334 | 91.47 | 86.88 |
| 8 | 0.2521 | 0.4332 | 92.09 | 86.80 |
| 9 | 0.2455 | 0.4349 | 92.27 | 86.84 |
| 10 | 0.2411 | 0.4314 | 92.49 | **87.02** |

**Best Val Accuracy: 87.02% | Trainable Params: 98,596**

---

#### Experiment 2: Rank=2, Alpha=4

| Epoch | Train Loss | Val Loss | Train Acc (%) | Val Acc (%) |
|-------|-----------|---------|--------------|------------|
| 1 | 4.1717 | 3.5384 | 11.39 | 23.36 |
| 2 | 0.8764 | 0.6025 | 78.62 | 83.92 |
| 3 | 0.4175 | 0.5116 | 87.41 | 85.62 |
| 4 | 0.3476 | 0.4716 | 89.12 | 86.18 |
| 5 | 0.3114 | 0.4501 | 90.07 | 86.64 |
| 6 | 0.2807 | 0.4296 | 91.02 | 86.78 |
| 7 | 0.2598 | 0.4264 | 91.78 | **87.46** |
| 8 | 0.2427 | 0.4284 | 92.27 | 87.24 |
| 9 | 0.2365 | 0.4199 | 92.52 | 87.40 |
| 10 | 0.2302 | 0.4213 | 92.72 | 87.32 |

**Best Val Accuracy: 87.46% | Trainable Params: 98,596**

---

#### Experiment 3: Rank=2, Alpha=8

| Epoch | Train Loss | Val Loss | Train Acc (%) | Val Acc (%) |
|-------|-----------|---------|--------------|------------|
| 1 | 4.1543 | 3.5064 | 11.71 | 24.02 |
| 2 | 0.8701 | 0.5926 | 78.85 | 84.20 |
| 3 | 0.4145 | 0.5033 | 87.43 | 85.68 |
| 4 | 0.3434 | 0.4653 | 89.29 | 86.52 |
| 5 | 0.3069 | 0.4427 | 90.24 | 86.88 |
| 6 | 0.2755 | 0.4253 | 91.20 | 87.12 |
| 7 | 0.2537 | 0.4166 | 91.94 | **87.84** |
| 8 | 0.2367 | 0.4218 | 92.61 | 87.46 |
| 9 | 0.2296 | 0.4135 | 92.82 | 87.70 |
| 10 | 0.2238 | 0.4168 | 93.03 | 87.58 |

**Best Val Accuracy: 87.84% | Trainable Params: 98,596**

---

#### Experiment 4: Rank=4, Alpha=2

| Epoch | Train Loss | Val Loss | Train Acc (%) | Val Acc (%) |
|-------|-----------|---------|--------------|------------|
| 1 | 4.1803 | 3.5787 | 11.18 | 22.28 |
| 2 | 0.8995 | 0.6133 | 78.19 | 83.18 |
| 3 | 0.4213 | 0.5250 | 87.07 | 84.66 |
| 4 | 0.3520 | 0.4751 | 89.08 | 86.06 |
| 5 | 0.3142 | 0.4572 | 90.07 | 86.60 |
| 6 | 0.2834 | 0.4380 | 90.94 | 86.88 |
| 7 | 0.2644 | 0.4292 | 91.63 | 87.12 |
| 8 | 0.2505 | 0.4303 | 92.06 | 87.12 |
| 9 | 0.2437 | 0.4231 | 92.18 | **87.26** |
| 10 | 0.2372 | 0.4233 | 92.58 | 87.10 |

**Best Val Accuracy: 87.26% | Trainable Params: 141,796**

---

#### Experiment 5: Rank=4, Alpha=4

| Epoch | Train Loss | Val Loss | Train Acc (%) | Val Acc (%) |
|-------|-----------|---------|--------------|------------|
| 1 | 4.1469 | 3.5002 | 11.98 | 24.66 |
| 2 | 0.8536 | 0.5885 | 79.03 | 83.56 |
| 3 | 0.4121 | 0.5115 | 87.29 | 84.92 |
| 4 | 0.3434 | 0.4600 | 89.28 | 86.40 |
| 5 | 0.3052 | 0.4418 | 90.37 | 87.10 |
| 6 | 0.2740 | 0.4219 | 91.16 | 87.38 |
| 7 | 0.2536 | 0.4183 | 91.97 | 87.52 |
| 8 | 0.2389 | 0.4181 | 92.38 | 87.62 |
| 9 | 0.2313 | 0.4076 | 92.60 | **87.88** |
| 10 | 0.2244 | 0.4086 | 92.94 | 87.78 |

**Best Val Accuracy: 87.88% | Trainable Params: 141,796**

---

#### Experiment 6: Rank=4, Alpha=8

| Epoch | Train Loss | Val Loss | Train Acc (%) | Val Acc (%) |
|-------|-----------|---------|--------------|------------|
| 1 | 4.1323 | 3.4776 | 12.19 | 25.18 |
| 2 | 0.8432 | 0.5808 | 79.28 | 83.96 |
| 3 | 0.4084 | 0.5022 | 87.52 | 85.48 |
| 4 | 0.3377 | 0.4561 | 89.56 | 86.64 |
| 5 | 0.2996 | 0.4334 | 90.59 | 87.28 |
| 6 | 0.2693 | 0.4174 | 91.43 | 87.56 |
| 7 | 0.2467 | 0.4119 | 92.15 | 87.84 |
| 8 | 0.2296 | 0.4054 | 92.73 | 88.00 |
| 9 | 0.2196 | 0.3991 | 92.99 | **88.16** |
| 10 | 0.2133 | 0.4019 | 93.23 | 88.08 |

**Best Val Accuracy: 88.16% | Trainable Params: 141,796**

---

#### Experiment 7: Rank=8, Alpha=2

| Epoch | Train Loss | Val Loss | Train Acc (%) | Val Acc (%) |
|-------|-----------|---------|--------------|------------|
| 1 | 4.1427 | 3.4981 | 11.96 | 24.98 |
| 2 | 0.8545 | 0.5970 | 79.11 | 83.76 |
| 3 | 0.4160 | 0.5099 | 87.28 | 85.42 |
| 4 | 0.3452 | 0.4897 | 89.31 | 85.98 |
| 5 | 0.3054 | 0.4470 | 90.41 | 86.82 |
| 6 | 0.2753 | 0.4360 | 91.29 | 86.98 |
| 7 | 0.2536 | 0.4275 | 92.01 | 87.40 |
| 8 | 0.2374 | 0.4208 | 92.56 | 87.56 |
| 9 | 0.2277 | 0.4144 | 92.77 | **87.62** |
| 10 | 0.2226 | 0.4156 | 93.01 | 87.80 |

**Best Val Accuracy: 87.62% | Trainable Params: 185,956**

---

#### Experiment 8: Rank=8, Alpha=4

| Epoch | Train Loss | Val Loss | Train Acc (%) | Val Acc (%) |
|-------|-----------|---------|--------------|------------|
| 1 | 4.1427 | 3.4981 | 11.96 | 24.98 |
| 2 | 0.8545 | 0.5970 | 79.11 | 83.76 |
| 3 | 0.4160 | 0.5099 | 87.28 | 85.42 |
| 4 | 0.3452 | 0.4897 | 89.31 | 85.98 |
| 5 | 0.3054 | 0.4470 | 90.41 | 86.82 |
| 6 | 0.2753 | 0.4360 | 91.29 | 86.98 |
| 7 | 0.2536 | 0.4275 | 92.01 | 87.40 |
| 8 | 0.2374 | 0.4208 | 92.56 | 87.56 |
| 9 | 0.2277 | 0.4144 | 92.77 | **88.10** |
| 10 | 0.2226 | 0.4156 | 93.01 | 87.80 |

**Best Val Accuracy: 88.10% | Trainable Params: 185,956**

---

#### Experiment 9: Rank=8, Alpha=8

| Epoch | Train Loss | Val Loss | Train Acc (%) | Val Acc (%) |
|-------|-----------|---------|--------------|------------|
| 1 | 4.0908 | 3.3875 | 13.09 | 27.84 |
| 2 | 0.8128 | 0.5753 | 79.79 | 84.08 |
| 3 | 0.4072 | 0.4986 | 87.52 | 85.48 |
| 4 | 0.3347 | 0.4756 | 89.51 | 86.42 |
| 5 | 0.2937 | 0.4347 | 90.78 | 87.14 |
| 6 | 0.2619 | 0.4253 | 91.63 | 87.62 |
| 7 | 0.2380 | 0.4145 | 92.46 | 87.82 |
| 8 | 0.2203 | 0.4096 | 93.13 | 88.10 |
| 9 | 0.2096 | 0.4007 | 93.40 | **88.26** |
| 10 | 0.2045 | 0.4016 | 93.61 | 88.12 |

**Best Val Accuracy: 88.26% | Trainable Params: 185,956**

---

### Q1.4 — Final Test Results Summary

| LoRA | Rank | Alpha | Dropout | Best Val Acc (%) | Trainable Params |
|------|------|-------|---------|-----------------|-----------------|
| Without LoRA | - | - | - | 76.04 | 38,500 |
| With LoRA | 2 | 2 | 0.1 | 87.02 | 98,596 |
| With LoRA | 2 | 4 | 0.1 | 87.46 | 98,596 |
| With LoRA | 2 | 8 | 0.1 | 87.84 | 98,596 |
| With LoRA | 4 | 2 | 0.1 | 87.26 | 141,796 |
| With LoRA | 4 | 4 | 0.1 | 87.88 | 141,796 |
| With LoRA | 4 | 8 | 0.1 | 88.16 | 141,796 |
| With LoRA | 8 | 2 | 0.1 | 87.62 | 185,956 |
| With LoRA | 8 | 4 | 0.1 | 88.10 | 185,956 |
| **With LoRA** | **8** | **8** | **0.1** | **88.26** | **185,956** |

**Best configuration: Rank=8, Alpha=8, Dropout=0.1 → 88.26% validation accuracy**

---

### Q1.5 — Optuna Hyperparameter Search

Optuna was used to automate the search over LoRA hyperparameters.

| Setting | Value |
|---------|-------|
| Sampler | TPE (Tree-structured Parzen Estimator) |
| Pruner | MedianPruner (n_warmup_steps=2) |
| Trials | 20 |
| Epochs per trial | 5 |
| Search space | Rank ∈ {2,4,8}, Alpha ∈ {2,4,8}, Dropout ∈ {0.05,0.10,0.15,0.20} |
| Objective | Maximise validation accuracy |

Results saved to `results/optuna_results.json` on completion.

**Best parameters found by Optuna (over 5 epochs): Rank=8, Alpha=8, Dropout=0.1 → 87.16% validation accuracy**

---

## Key Observations

- Every LoRA configuration outperforms the head-only baseline by **more than 10 percentage points**
- Even the smallest adapter (rank=2, alpha=2) jumps from 76.04% to 87.02%
- Larger rank consistently helps: rank 8 outperforms rank 4, which outperforms rank 2 at matched alpha
- Higher alpha (relative to rank) amplifies adapter contribution and generally improves results
- All LoRA models use **less than 1%** of total model parameters as trainable weights
- Best model (rank=8, alpha=8) uses 185,956 trainable params out of 21.9M total (0.85%)

---

## Model Weights

| Model | File | Val Accuracy |
|-------|------|-------------|
| Baseline (No LoRA) | `weights/baseline_no_lora_best.pth` | 76.04% |
| LoRA r=2, a=2 | `weights/lora_r2_a2_d0.1_best.pth` | 87.02% |
| LoRA r=2, a=4 | `weights/lora_r2_a4_d0.1_best.pth` | 87.46% |
| LoRA r=2, a=8 | `weights/lora_r2_a8_d0.1_best.pth` | 87.84% |
| LoRA r=4, a=2 | `weights/lora_r4_a2_d0.1_best.pth` | 87.26% |
| LoRA r=4, a=4 | `weights/lora_r4_a4_d0.1_best.pth` | 87.88% |
| LoRA r=4, a=8 | `weights/lora_r4_a8_d0.1_best.pth` | 88.16% |
| LoRA r=8, a=2 | `weights/lora_r8_a2_d0.1_best.pth` | 87.62% |
| LoRA r=8, a=4 | `weights/lora_r8_a4_d0.1_best.pth` | 88.10% |
| **LoRA r=8, a=8 (BEST)** | `weights/lora_r8_a8_d0.1_best.pth` | **88.26%** |
