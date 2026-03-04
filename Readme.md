```markdown
# End-to-End Hugging Face Model Training and Docker Deployment

## Project Overview

This project demonstrates a complete machine learning workflow using the Hugging Face ecosystem and Docker. The objective of the assignment is to train a text classification model, evaluate its performance, upload the trained model to the Hugging Face Hub, and run evaluation inside a Docker container.

---

## Project Objectives

The main objectives of this project are:

- Convert notebook code into reusable Python scripts
- Train a transformer-based model using Hugging Face
- Evaluate the trained model
- Upload the model to Hugging Face Hub
- Re-evaluate the model by loading it from Hugging Face
- Create a Docker container that automatically runs evaluation
- Publish the complete project on GitHub

---

## Model Details

**Model Used:** DistilBERT

DistilBERT is a lightweight and faster version of the BERT model designed for efficient natural language processing tasks.

### Why DistilBERT?

- Smaller model size
- Faster training and inference
- Strong performance on NLP classification tasks
- Fully supported by Hugging Face Transformers

## Dataset

The dataset used in this project comes from **Goodreads review data** across multiple genres.

Genres included in the dataset:

- Poetry
- Romance
- Fantasy
- Mystery / Thriller
- History / Biography
- Young Adult
- Comics
- Children

### Dataset Processing Steps

1. Download review datasets from multiple genre sources
2. Extract review text
3. Clean and filter the text data
4. Split dataset into training and test sets
5. Convert genre labels into numerical labels
6. Tokenize text using DistilBERT tokenizer

---

## Project Structure

```

DLOPS_Assignment-3
│
├── data/
│   ├── train.json
│   ├── test.json
│
├── src/
│   ├── data.py
│   ├── train.py
│   ├── evaluate.py
│   ├── hf.py
│
├── models/
│   └── hf_model.py
│
├── distilbert-reviews-genres-lora/
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── training_args.bin
│
├── Dockerfile
├── requirements.txt
├── README.md
└── report.pdf

````

---

## Training the Model

To train the model locally:

```bash
python train.py
````

The training process will:

* Load the DistilBERT model
* Apply LoRA adapters
* Train the model using Hugging Face Trainer
* Save the trained model

---

## Model Evaluation

To evaluate the model locally:

```bash
python evaluate.py
```

Evaluation metrics include:

* Accuracy
* Loss

---

## Hugging Face Model

The trained model is uploaded to the Hugging Face Hub.

**Model Link**

[https://huggingface.co/anchitya/book-genre-classifier](https://huggingface.co/anchitya/book-genre-classifier)

The repository contains:

* Model weights
* Tokenizer files
* Configuration files
* Training metadata

---

## Docker Deployment

A Docker container was created to run the evaluation automatically.

The Docker container performs the following tasks:

1. Install required Python dependencies
2. Download the model from Hugging Face
3. Load the evaluation dataset
4. Run the evaluation script

---

## Docker Image (Docker Hub)

The Docker image for this project is available on Docker Hub.

**Docker Hub Repository**

```
ADD_DOCKER_LINK_HERE
```

Example:

```
https://hub.docker.com/r/YOUR_USERNAME/dlops-eval
```

### Pull Docker Image

```bash
docker pull YOUR_USERNAME/dlops-eval
```

### Run Docker Container

```bash
docker run YOUR_USERNAME/dlops-eval
```

---

## Build Docker Image Locally

```bash
docker build -t dlops-eval .
```

---

## Run Docker Container Locally

```bash
docker run dlops-eval
```

The container will automatically:

* Download the model from Hugging Face
* Load the evaluation dataset
* Run evaluation

---

## Example Evaluation Output

```
Evaluation Results

Loss: 0.72
Accuracy: 0.86
Runtime: 451 seconds
```

---

## Challenges Faced

During the implementation of this project, several challenges were encountered:

* Managing LoRA adapter weights
* Correctly merging LoRA weights with the base model
* Uploading the model to Hugging Face with the correct file structure
* Debugging Docker container dependency issues

These challenges were resolved through iterative debugging and testing.

---


