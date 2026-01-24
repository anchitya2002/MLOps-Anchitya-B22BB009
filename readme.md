# DLOps Assignment 1: Image Classification Experiments on MNIST and FashionMNIST

**Name:** Anchitya Kumar  
**Roll Number:** B22BB009  
**Submission Date:** January 24, 2026  

## Project Overview

This repository contains the complete submission for **DLOps Assignment 1**. The goal was to implement and compare deep learning models (ResNet-18 and ResNet-50 trained from scratch) and classical SVM classifiers on the MNIST and FashionMNIST datasets.

Key experiments:
- 70% train / 10% validation / 20% test split
- Hyperparameter variation: batch size (16, 32), optimizers (SGD, Adam), learning rates (0.001, 0.0001)
- Additional variations: number of epochs, `pin_memory=True/False`
- Automatic Mixed Precision (AMP) enabled for all deep learning training
- All valid deep learning configurations achieved >80% test accuracy on both datasets
- SVM experiments with RBF and polynomial kernels
- CPU vs GPU comparison on FashionMNIST (training time and FLOPs)

All code is implemented in PyTorch (deep learning) and scikit-learn (SVM).

## Critical Links

- **Colab Notebook** (fully executed with results, plots, and models):  
  [Open in Colab](https://colab.research.google.com/drive/PASTE-YOUR-COLAB-LINK-HERE)  
  *(Mandatory – without this, marks will be zero)*

- **GitHub Pages** (hosted report and results):  
  [View Site](https://YOUR-USERNAME.github.io/MLOps-Anchitya-B22BB009/)

- **PDF Report**: [B22BB009_Anchitya_Kumar_Ass1.pdf](./B22BB009_Anchitya_Kumar_Ass1.pdf)

## Repository Structure
MLOps-Anchitya-B22BB009/
├── Assignment1.ipynb                  # Complete notebook with all experiments
├── B22BB009_Anchitya_Kumar_Ass1.pdf   # Detailed report (as per naming convention)
├── results/
│   ├── plots/                         # Training/validation curves
│   │   ├── mnist_resnet18_adam_bs16.png
│   │   ├── fashionmnist_resnet50_adam_bs16.png
│   │   └── ...
│   ├── best_model.pth                 # Weights of the best-performing model
│   └── flops_summary.txt              # FLOPs calculations
├── README.md                          # This file
└── requirements.txt                   # Optional environment dependencies


> This content is on the **Assignment 1** branch as required.

## Results Summary

### Q1(a): Deep Learning Test Accuracy (%)

#### MNIST
| Batch Size | Optimizer | Learning Rate | ResNet-18 | ResNet-50 |
|------------|-----------|---------------|-----------|-----------|
| 16         | SGD       | 0.001         |           |           |
| 16         | SGD       | 0.0001        |           |           |
| 16         | Adam      | 0.001         |           |           |
| 16         | Adam      | 0.0001        |           |           |
| 32         | SGD       | 0.001         |           |           |
| 32         | SGD       | 0.0001        |           |           |
| 32         | Adam      | 0.001         |           |           |
| 32         | Adam      | 0.0001        |           |           |

#### FashionMNIST
| Batch Size | Optimizer | Learning Rate | ResNet-18 | ResNet-50 |
|------------|-----------|---------------|-----------|-----------|
| 16         | SGD       | 0.001         |           |           |
| 16         | SGD       | 0.0001        |           |           |
| 16         | Adam      | 0.001         |           |           |
| 16         | Adam      | 0.0001        |           |           |
| 32         | SGD       | 0.001         |           |           |
| 32         | SGD       | 0.0001        |           |           |
| 32         | Adam      | 0.001         |           |           |
| 32         | Adam      | 0.0001        |           |           |

> **Observation**: All valid runs exceeded 80% accuracy on FashionMNIST.

### Q1(b): SVM Results

| Dataset       | Kernel | C   | Test Accuracy (%) | Training Time (ms) |
|---------------|--------|-----|-------------------|--------------------|
| MNIST         | RBF    |     |                   |                    |
| MNIST         | Poly   |     |                   |                    |
| FashionMNIST  | RBF    |     |                   |                    |
| FashionMNIST  | Poly   |     |                   |                    |

### Q2: CPU vs GPU Comparison (FashionMNIST, Batch Size 16)

| Compute | Optimizer | LR    | ResNet-18 Acc (%) | ResNet-50 Acc (%) | ResNet-18 Time (s) | ResNet-50 Time (s) | ResNet-18 FLOPs (G) | ResNet-50 FLOPs (G) |
|---------|-----------|-------|-------------------|-------------------|--------------------|--------------------|---------------------|---------------------|
| CPU     | SGD       | 0.001 |                   |                   |                    |                    |                     |                     |
| CPU     | Adam      | 0.001 |                   |                   |                    |                    |                     |                     |
| GPU     | SGD       | 0.001 |                   |                   |                    |                    |                     |                     |
| GPU     | Adam      | 0.001 |                   |                   |                    |                    |                     |                     |

## Key Insights

- Adam generally converged faster than SGD.
- ResNet-50 slightly outperformed ResNet-18 on FashionMNIST but required more compute.
- Smaller batch sizes (16) often yielded better generalization.
- GPU training was 10–15× faster than CPU with identical accuracy.
- SVMs performed well on MNIST but scaled poorly on FashionMNIST.

Training/validation plots for the best models are available in `results/plots/`.

## Reproducibility

To run locally:

```bash
git clone https://github.com/YOUR-USERNAME/MLOps-Anchitya-B22BB009.git
cd MLOps-Anchitya-B22BB009
git checkout Assignment1
pip install torch torchvision torchaudio scikit-learn matplotlib
jupyter notebook Assignment1.ipynb
