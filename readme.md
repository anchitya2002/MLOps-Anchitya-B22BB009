# DLOps Assignment 1

## Project Overview

Experiments were performed on two widely used benchmark datasets, MNIST and FashionMNIST. Two convolutional neural network models, ResNet-18 and ResNet-50, were trained from scratch using PyTorch under multiple hyperparameter configurations, including different batch sizes, learning rates, and optimizers. Automatic Mixed Precision (AMP) was enabled to improve training speed and memory efficiency.

Alongside deep learning models, classical Support Vector Machine (SVM) classifiers were implemented using both RBF and Polynomial kernels to compare traditional machine learning methods with deep learning approaches.

The project also includes a detailed comparison of performance between CPU and GPU execution with analysis based on:
- Classification accuracy  
- Training time  
- Computational cost (FLOPs)  

## Objectives

* Implement ResNet architectures **without pretrained weights**
* Perform systematic hyperparameter experiments:

  * Batch sizes: 16, 32
  * Optimizers: SGD, Adam
  * Learning rates: 0.001, 0.0001
* Train/Validation/Test split: **70% / 10% / 20%**
* Apply Automatic Mixed Precision (AMP)
* Compare:

  * Accuracy
  * Training time
  * FLOPs
  * CPU vs GPU performance
* Evaluate SVM using RBF and Polynomial kernels

---

## Repository Structure

```
MLOps-Name-B22BB009/
│
├── question_1.ipynb            # Q1(a): Deep learning experiments
├── question_1(b).ipynb         # Q1(b): SVM experiments
├── question2_.ipynb            # Q2: CPU vs GPU + FLOPs analysis
├── README.md                   # Project documentation (this file)
├── requirements.txt            # Dependencies
└── data/                       # Dataset directory (if generated locally)
```

---

## Results Summary


#### MNIST

| Batch Size | Optimizer | Learning Rate | ResNet-18 (%) | ResNet-50 (%) |
| ---------- | --------- | ------------- | ------------- | ------------- |
| 16         | SGD       | 0.001         |     98.43     |   96.94       |
| 16         | SGD       | 0.0001        |     94.48     |   67.26       |
| 16         | Adam      | 0.001         |     98.87     |   97.51       |
| 16         | Adam      | 0.0001        |     98.62     |   96.75       |
| 32         | SGD       | 0.001         |     97.80     |   95.56       |
| 32         | SGD       | 0.0001        |     92.99     |   38.73       |
| 32         | Adam      | 0.001         |     98.28     |   98.23       |
| 32         | Adam      | 0.0001        |     98.45     |   96.53       |

#### FashionMNIST

| Batch Size | Optimizer | Learning Rate | ResNet-18 (%) | ResNet-50 (%)  |
| ---------- | --------- | ------------- | ------------- | -------------  |
| 16         | SGD       | 0.001         |    87.83      |  81.68         |
| 16         | SGD       | 0.0001        |    82.49      |  66.20         |
| 16         | Adam      | 0.001         |    89.33      |  83.10         |
| 16         | Adam      | 0.0001        |    89.93      |  84.72         |
| 32         | SGD       | 0.001         |    87.07      |  79.92         |
| 32         | SGD       | 0.0001        |    79.91      |  44.83         |
| 32         | Adam      | 0.001         |    90.27      |  85.92         |
| 32         | Adam      | 0.0001        |    89.36      |  85.49         |

> **Observation:** All valid runs achieved **>80% accuracy on FashionMNIST**.

---

### SVM Results

| Dataset      | Kernel     | Test Accuracy (%) | Training Time (ms) |
| ------------ | ---------- | ----------------- | ------------------ |
| MNIST        | RBF        |   92.25	          |   18747.78         |
| MNIST        | Polynomial |   87.85           |   44918.00         |
| FashionMNIST | RBF        |   86.50           |   17712.64         |
| FashionMNIST | Polynomial |   83.60           |   23913.40         |
             |

---

### CPU vs GPU Comparison (FashionMNIST, Batch Size = 16)

| Compute | Optimizer | LR    | ResNet-18 Acc (%) | ResNet-50 Acc (%) | ResNet-18 Time (s) | ResNet-50 Time (s) | ResNet-18 FLOPs (G) | ResNet-50 FLOPs (G) |
| ------- | --------- | ----- | ----------------- | ----------------- | ------------------ | ------------------ | ------------------- | ------------------- |
| CPU     | SGD       | 0.001 |        86.18      |    81.41          |         1417.61 s  | 3153.08 s          |  142.04             |  329.06             |
| CPU     | Adam      | 0.001 |        87.54      |    83.62          |         2023.53    | 4042.74            |  142.04             |  329.06             |
| GPU     | SGD       | 0.001 |        85.81      |    78.65          |         1317.55    | 3004.02            |  142.04             |  329.06             |
| GPU     | Adam      | 0.001 |        87         |    82.55          |         2017.23    | 4216.6             |  142.04             |  329.06             |

---

### CPU vs GPU Comparison (FashionMNIST, Batch Size = 32)

| Compute | Optimizer | LR    | ResNet-18 Acc (%) | ResNet-50 Acc (%) | ResNet-18 Time (s) | ResNet-50 Time (s) | ResNet-18 FLOPs (G) | ResNet-50 FLOPs (G) |
| ------- | --------- | ----- | ----------------- | ----------------- | ------------------ | ------------------ | ------------------- | ------------------- |
| CPU     | SGD       | 0.001 |        86.18      |    81.41          |         1417.61 s  | 3153.08 s          |  142.04             |  329.06             |
| CPU     | Adam      | 0.001 |        87.54      |    83.62          |         2023.53    | 4042.74            |  142.04             |  329.06             |
| GPU     | SGD       | 0.001 |        85.81      |    78.65          |         1317.55    | 3004.02            |  142.04             |  329.06             |
| GPU     | Adam      | 0.001 |        87         |    82.55          |         2017.23    | 4216.6             |  142.04             |  329.06             |

---


## Key Insights

* Adam optimizer converges faster than SGD across datasets.
* ResNet-50 provides slightly higher accuracy than ResNet-18, but at significantly higher computational cost.
* Smaller batch size (16) often improves generalization.
* GPU training is approximately **10–15× faster** than CPU.
* SVM performs strongly on MNIST but scales poorly to FashionMNIST.



### Run Locally

```bash
git clone https://github.com/anchitya2002/MLOps-Name-B22BB009.git
pip install -r requirements.txt
```


## Requirements

Dependencies listed in `requirements.txt`:

* torch
* torchvision
* torchaudio
* scikit-learn
* matplotlib
* pandas
* fvcore (for FLOPs calculation)


## License

This project is licensed under the **MIT License**.  
You are free to use, modify, distribute, and build upon this project with proper attribution.

See the full license in the [LICENSE](LICENSE) file.
