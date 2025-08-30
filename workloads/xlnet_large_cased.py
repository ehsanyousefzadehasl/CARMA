# Import necessary libraries
import os
import datasets
import torch
from transformers import XLNetTokenizer, XLNetForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding

from modelsummary import summary  # Import modelsummary

import time


start = time.time()

# Set the environment variable to suppress unnecessary logs
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# Step 1: Load the Wiki dataset
# You can use any wiki dataset available in Hugging Face datasets library.
# Here, we use a simplified version of the WikiText dataset for demonstration.

# Load the WikiText dataset
raw_datasets = datasets.load_dataset("wikitext", "wikitext-2-raw-v1")

# Check if data is loaded correctly
print(f"Available splits in the dataset: {raw_datasets.keys()}")

# Step 2: Initialize the tokenizer
tokenizer = XLNetTokenizer.from_pretrained("xlnet-large-cased", timeout=60)

# Tokenization function
def tokenize_function(examples):
    # Tokenize the text and create a dummy label
    result = tokenizer(examples["text"], truncation=True, max_length=512)
    result["labels"] = [1] * len(result["input_ids"])  # Dummy label
    return result

# Apply tokenization to the dataset
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

# Check the tokenized dataset keys
print(f"Available splits in the tokenized dataset: {tokenized_datasets.keys()}")

# Step 3: Set up the data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Step 4: Define training arguments
training_args = TrainingArguments(
    output_dir="./xlnet-wiki-output",  # This directory is unused
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    evaluation_strategy="no",  # Disable evaluation logging
    save_strategy="no",        # Disable checkpoint saving
    logging_strategy="no",     # Disable logging
    save_total_limit=None,     # No limit on saving checkpoints
    report_to=[],              # No reporting to any logger
    do_train=True,             # Enable training
    do_eval=False,             # Disable evaluation to minimize outputs
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Step 5: Initialize the model
model = XLNetForSequenceClassification.from_pretrained("xlnet-large-cased").to(device)


#====
# Create a mock input tensor for the model
batch_size = 4      # Define your batch size
seq_length = 512    # Define sequence length (up to `max_position_embeddings`)
mock_input = torch.randint(0, model.config.vocab_size, (batch_size, seq_length), dtype=torch.long).to(device)

# Check model compatibility with the mock input
try:
    model(mock_input)  # Test with the mock input
    print("Dummy input works with the model.")
except Exception as e:
    print(f"Error during dummy input check: {e}")

# Generate model summary
try:
    summary(model, mock_input, show_input=True, show_hierarchical=True)
except Exception as e:
    print(f"Error during summary: {e}")
#====


# Step 6: Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Step 7: Train the model
trainer.train()

# Step 8: Evaluate the model
results = trainer.evaluate()

# Print evaluation results
print(f"Evaluation results: {results}")

end = time.time()

execution_time = end - start

print("\nexecution time: ", execution_time)