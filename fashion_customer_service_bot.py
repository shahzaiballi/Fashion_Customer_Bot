import pandas as pd
import torch
from transformers import GPT2Tokenizer, DistilGPT2LMHeadModel, Trainer, TrainingArguments
from datasets import Dataset
import os

# 1. Set up the environment
# Ensure you have installed: pip install transformers datasets pandas torch

# 2. Prepare and preprocess the dataset
# Create a mock dataset (save this as FashionHelpQA.csv in your project folder)
data = [
    {"query_id": 1, "customer_query": "Hi, I want to know the status of my order #12345.", 
     "ideal_response": "To check your order status, please visit your account page at [link] or provide your email address so I can look it up for you."},
    {"query_id": 2, "customer_query": "What is your return policy for sale items?", 
     "ideal_response": "Sale items can be returned for store credit within 14 days of receipt, provided they are unworn and have all tags attached. For full details, please see our policy here: [link]."},
    {"query_id": 3, "customer_query": "Do you have this dress in blue? Product ID: XYZ789", 
     "ideal_response": "Let me check that for you. It appears the XYZ789 dress is currently out of stock in blue. Would you like me to notify you when it's back in stock, or perhaps I can suggest a similar style available in blue?"},
    {"query_id": 4, "customer_query": "How do I care for a silk blouse?", 
     "ideal_response": "For silk blouses, we recommend gentle hand washing in cold water with a mild detergent, or dry cleaning. Avoid wringing and lay flat to dry away from direct sunlight."},
    {"query_id": 5, "customer_query": "I received the wrong size for my shoes. Order #67890", 
     "ideal_response": "I'm so sorry to hear that you received the wrong size for order #67890! Please provide the item name and the size you received, and the size you originally ordered. We'll arrange an exchange for you right away."}
]

# Save dataset to CSV
df = pd.DataFrame(data)
df.to_csv("FashionHelpQA.csv", index=False)

# Load and preprocess dataset
df = pd.read_csv("FashionHelpQA.csv")
# Format for DistilGPT-2: combine query and response into a single text with a separator
df['text'] = df['customer_query'] + " ### " + df['ideal_response']
dataset = Dataset.from_pandas(df[['text']])

# Load tokenizer (using GPT2Tokenizer for DistilGPT-2)
tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
tokenizer.pad_token = tokenizer.eos_token  # Set padding token

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=128)

tokenized_dataset = dataset.map(tokenize_function, batched=True)
# Split into train (80%) and validation (20%)
train_size = int(0.8 * len(tokenized_dataset))
train_dataset = tokenized_dataset.select(range(train_size))
eval_dataset = tokenized_dataset.select(range(train_size, len(tokenized_dataset)))

# 3. Fine-tune the model
# Load model
model = DistilGPT2LMHeadModel.from_pretrained("distilgpt2")

# Define training arguments (optimized for low memory)
training_args = TrainingArguments(
    output_dir="./fashion_bot_results",
    num_train_epochs=3,  # Small number of epochs for quick training
    per_device_train_batch_size=2,  # Small batch size for 8GB RAM
    per_device_eval_batch_size=2,
    warmup_steps=10,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# Train the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./fashion_bot_model")
tokenizer.save_pretrained("./fashion_bot_model")

# 4. Evaluate the model
# Function to generate response
def generate_response(query):
    inputs = tokenizer(query + " ### ", return_tensors="pt", padding=True, truncation=True, max_length=128)
    outputs = model.generate(**inputs, max_length=200, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split(" ### ")[1] if " ### " in response else response

# Test with sample queries
sample_queries = [
    "What is the status of my order #54321?",
    "Can I return a sale item after 10 days?",
    "Do you have this jacket in red? Product ID: ABC123"
]

for query in sample_queries:
    print(f"Query: {query}")
    print(f"Response: {generate_response(query)}\n")
