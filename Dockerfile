FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .

# Install PyTorch with CUDA 11.8 support (compatible with CUDA 11.4 driver)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY data.py model.py train.py test.py utils.py ./
COPY run_all_experiments.py optuna_search.py upload_hf.py ./
COPY configs/ ./configs/

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV WANDB_DIR=/app/wandb_logs

# Create directories for outputs
RUN mkdir -p /app/weights /app/results /app/wandb_logs /app/data

# Default command
CMD ["python", "run_all_experiments.py", "--epochs", "10", "--batch_size", "16", "--num_workers", "2"]
