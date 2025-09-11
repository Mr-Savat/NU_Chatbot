from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Use the absolute path is more reliable, especially when running from different directories.
# Example for Windows: MODEL_PATH = r"C:\Users\User\OneDrive\Documents\projectAi\backend\gpt2-norton"
MODEL_PATH = "./gpt2-norton"

# Load the model and tokenizer from the local directory
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = GPT2LMHeadModel.from_pretrained(MODEL_PATH, local_files_only=True)

# Set the padding token (this should already be saved in your tokenizer config, but it's safe to do again)
tokenizer.pad_token = tokenizer.eos_token

# Set the model to evaluation mode for inference (important!)
model.eval()

def ask_gpt2(question, max_length=150):
    """
    Generates a response to the given question using the fine-tuned GPT-2 model.

    Args:
        question (str): The user's input/question.
        max_length (int): The maximum length of the generated text.

    Returns:
        str: The generated answer.
    """
    # Encode the input, add the model's required parameters (like attention_mask)
    inputs = tokenizer.encode(question, return_tensors="pt")
    
    # Generate a response
    with torch.no_grad():  # Disable gradient calculation for inference (faster, uses less memory)
        outputs = model.generate(
            inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,  # Controls randomness: lower = more deterministic, higher = more random
            do_sample=True,   # Sample from the probability distribution instead of greedy decoding
            pad_token_id=tokenizer.eos_token_id, # Important to avoid pad token warnings
            attention_mask=inputs.ne(tokenizer.pad_token_id).long() if tokenizer.pad_token_id is not None else None
        )
    
    # Decode the generated tokens back to text and remove the input question
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # A simple way to try and get only the new part of the answer
    answer = full_response.replace(question, "").strip()
    
    return answer if answer else full_response # Fallback to the full response if parsing fails