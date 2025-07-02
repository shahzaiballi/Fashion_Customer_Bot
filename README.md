# 🛍️ Fashion Customer Support Bot (Fine-Tuned DistilGPT-2)

This project demonstrates how to fine-tune [DistilGPT-2](https://huggingface.co/distilgpt2) on a small dataset of fashion-related customer service queries using the 🤗 Hugging Face `transformers`, `datasets`, and `Trainer` APIs. The goal is to build a chatbot capable of responding to fashion e-commerce support queries in a helpful, conversational tone.

---

## 📂 Project Structure

```
Fashion_Customer_Bot/
├── FashionHelpQA.csv           # Custom dataset for training
├── fashion_bot_model/          # Folder containing the fine-tuned model and tokenizer
├── fashion_bot_results/        # Output directory with checkpoints and logs
├── logs/                       # Training logs
├── main.py                     # Main script for training and evaluation
└── README.md                   # This file
```

---

## 💡 Features

- Fine-tunes DistilGPT-2 on a domain-specific fashion Q&A dataset.
- Simple conversational format using `###` as a separator between queries and responses.
- Supports generation of AI responses to new customer queries.
- Lightweight training setup for 8GB RAM GPUs or Colab environments.

---

## 🛠️ Requirements

Make sure the following Python libraries are installed:

```bash
pip install pandas torch transformers datasets
```

Tested with:
- Python 3.8+
- PyTorch 1.13+
- Hugging Face Transformers 4.x

---

## 📊 Dataset

A mock dataset (`FashionHelpQA.csv`) is used with columns:

- `query_id`
- `customer_query`
- `ideal_response`

Each row represents a sample query and its appropriate support response.

---

## 🚀 How to Run

1. Clone this repository or copy the script into your working directory.

2. Prepare the dataset:

```python
# Save the mock data
df = pd.DataFrame(data)
df.to_csv("FashionHelpQA.csv", index=False)
```

3. Fine-tune the model:

```bash
python main.py
```

4. Generate predictions:

The script includes a simple test function at the end:
```python
generate_response("What is the status of my order #54321?")
```

---

## 🧠 Model Details

- Base Model: `distilgpt2` (smaller version of GPT-2)
- Tokenizer: `GPT2Tokenizer`
- Max input length: 128 tokens
- Response generation: Up to 200 tokens

---

## 🗂️ Training Configuration

```python
TrainingArguments(
    output_dir="./fashion_bot_results",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True
)
```

---

## 🧪 Sample Results

```
Query: What is the status of my order #54321?
Response: To check your order status, please visit your account page at [link] or provide your email address so I can look it up for you.

Query: Can I return a sale item after 10 days?
Response: Sale items can be returned for store credit within 14 days of receipt, provided they are unworn and have all tags attached.

Query: Do you have this jacket in red? Product ID: ABC123
Response: Let me check that for you. It appears the ABC123 jacket is currently out of stock in red. Would you like me to notify you when it's back?
```

---

## 📦 Output

After training, the model and tokenizer are saved to:

```
./fashion_bot_model/
├── config.json
├── pytorch_model.bin
├── tokenizer.json
├── tokenizer_config.json
└── vocab.json
```

---

## 📜 License

This project is for educational purposes. The underlying models are provided by [Hugging Face](https://huggingface.co/) under their respective licenses.

---

## 🤝 Contributions

Pull requests and suggestions are welcome! If you'd like to improve this bot, feel free to fork or raise an issue.

---

## ✨ Author

**Shahzaib Ali** – _AI/ML & Software Engineer_
