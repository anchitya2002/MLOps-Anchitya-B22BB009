# Assignment 4 — Transformer Translation (Ray Tune + Optuna)

**Anchitya Kumar (B22BB009)**
IIT Jodhpur

---

## Overview

This project improves an English → Hindi Transformer model using **hyperparameter tuning**.

* Baseline model: trained for 100 epochs with fixed settings
* Tuned model: optimized using **Ray Tune + Optuna**
* Achieved better performance in **much less time**

---

## Results

| Metric        | Baseline | Tuned Model |
| ------------- | -------- | ----------- |
| Epochs        | 100      | 15          |
| Training Time | ~67 min  | 7.6 min     |
| Final Loss    | 0.0980   | 0.4062      |
| BLEU Score    | 51.11    | **73.79**   |

---

## Best Hyperparameters

```
Learning Rate     = 3.56e-4
Batch Size        = 32
Attention Heads   = 4
FFN Dimension     = 2048
Dropout           = 0.10
LR Scheduler      = Cosine Annealing
```

---

## How to Run

### Install dependencies

```
pip install "ray[train,tune]" optuna torch nltk pandas tqdm
```

### Run notebooks

* Baseline: `en_to_hi.ipynb`
* Tuned: `b22bb009_ass_4_tuned_en_to_hi.ipynb`

---

Final model achieved **73.79 BLEU in just 15 epochs**.
