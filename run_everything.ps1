$env:WANDB_API_KEY="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"
$env:WANDB_MODE="online"

echo "=========================================================="
echo "Starting All Experiments (Baseline + 9 LoRA combinations)"
echo "=========================================================="

docker run --gpus all --name dlops-run-experiments -v "$pwd\weights:/app/weights" -v "$pwd\results:/app/results" -v "$pwd\data:/app/data" -e WANDB_API_KEY="$env:WANDB_API_KEY" -e WANDB_MODE="online" dlops-assignment5 python run_all_experiments.py --epochs 10 --batch_size 16 --num_workers 2

echo "=========================================================="
echo "Starting Optuna Hyperparameter Search"
echo "=========================================================="

docker run --gpus all --name dlops-run-optuna -v "$pwd\weights:/app/weights" -v "$pwd\results:/app/results" -v "$pwd\data:/app/data" -e WANDB_API_KEY="$env:WANDB_API_KEY" -e WANDB_MODE="online" dlops-assignment5 python optuna_search.py --n_trials 20 --epochs 5 --batch_size 16 --num_workers 2

echo "=========================================================="
echo "Experiments and Optuna Search Completed!"
echo "Next step: Provide HuggingFace Token to upload the model."
echo "=========================================================="
