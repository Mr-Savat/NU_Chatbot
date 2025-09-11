from datasets import load_dataset
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
import os

# Paths
TRAIN_FILE = "./faqs.txt"  # Consider renaming your data file
OUTPUT_DIR = "./gpt2-norton"

# Load GPT-2 tokenizer & model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token # Set padding token
model = GPT2LMHeadModel.from_pretrained("gpt2")

# --- CRITICAL: Format your data for Q&A ---
# Your train.txt should format examples with a separator.
# Example format for each line:
# "Question: What is Norton University? Answer: It is a private university in Cambodia.<|endoftext|>"
# This teaches the model the pattern you want.

def format_dataset(file_path):
    """Reads the raw text file and ensures each example is on a new line."""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    return lines

# Load and format dataset
dataset = load_dataset("text", data_files={"train": TRAIN_FILE})

# Tokenize dataset
def tokenize_function(examples):
    # Tokenize without manual padding. The data collator will handle dynamic padding during training.
    return tokenizer(examples["text"], truncation=True, max_length=256)

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Data collator for dynamic padding (more efficient than pre-padding all sequences)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False, # We are doing Causal LM, not Masked LM
)

# Training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=10,          # Increased slightly for a small dataset
    per_device_train_batch_size=4, # Increased batch size if your GPU can handle it
    gradient_accumulation_steps=2,  # Simulates a larger batch size if needed
    save_steps=500,
    save_total_limit=2,
    logging_steps=50,
    prediction_loss_only=True,
    fp16=True,  # Use if you have a compatible GPU (faster training)
    dataloader_pin_memory=False, # Can help with speed on some systems
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# Train
print("Starting training...")
trainer.train()

# Save the final model and tokenizer
trainer.save_model()
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"✅ Fine-tuning complete! Model saved to {OUTPUT_DIR}")