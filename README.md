# Fashion Customer Service Bot

This project creates a simple chatbot for a fashion e-commerce store. It uses a lightweight **DistilGPT-2** model to answer customer questions about orders, returns, products, and styling advice. The code is designed to run on a laptop with 8GB RAM.

## What It Does
- Fine-tunes DistilGPT-2 on a small dataset of customer queries and responses.
- Answers questions like "What's my order status?" or "Can I return sale items?"
- Saves the trained model and tests it with sample questions.

## Requirements
- Python 3.8 or higher
- 8GB RAM laptop (CPU only)
- Internet connection for installing libraries

## Setup
1. Clone or download this project:
   ```bash
   git clone https://github.com/your-username/fashion-customer-service-bot.git
   cd fashion-customer-service-bot
   ```

2. Create a virtual environment:
   ```bash
   python -m venv env
   .\env\Scripts\activate  # Windows
   source env/bin/activate  # Linux/Mac
   ```

3. Install libraries:
   ```bash
   pip install transformers datasets pandas torch
   ```

## Dataset
- **File**: `FashionHelpQA.csv`
- Contains customer questions and ideal answers.
- Example:
  ```csv
  query_id,customer_query,ideal_response
  1,"Hi, I want to know the status of my order #12345.","To check your order status, please visit your account page at [link] or provide your email address so I can look it up for you."
  ```
- The script creates a small dataset (5 rows). Add more rows (500-1000) for better results.

## How to Run
1. Ensure `FashionHelpQA.csv` is in the project folder (created by the script if missing).
2. Run the script:
   ```bash
   python fashion_customer_service_bot.py
   ```
3. What happens:
   - Creates/loads the dataset.
   - Trains the model (~10-20 minutes on 8GB RAM).
   - Saves the model to `./fashion_bot_model`.
   - Shows example responses like:
     ```
     Query: What is the status of my order #54321?
     Response: To check your order status, please visit your account page at [link]...
     ```

