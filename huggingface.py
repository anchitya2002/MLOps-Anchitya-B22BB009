from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel
from huggingface_hub import upload_folder

# Path to LoRA adapter folder
lora_path = r"C:\Users\anchi\OneDrive\Documents\DLOPS_Assignement-3\distilbert-reviews-genres-lora"

# Load base model
base_model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-cased",
    num_labels=8
)

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, lora_path)

# Merge LoRA weights into base model
model = model.merge_and_unload()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")

# HuggingFace repo
repo_name = "anchitya/book-genre-classifier"

# Push merged model + tokenizer
model.push_to_hub(repo_name)
tokenizer.push_to_hub(repo_name)

# Upload entire LoRA folder (all files)
upload_folder(
    folder_path=lora_path,
    repo_id=repo_name,
    repo_type="model"
)