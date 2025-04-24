


# models.py
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import os

# Custom Dataset class for email classification
class EmailDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Training function
def train_model():
    print("Starting model training...")
    # Check if dataset exists
    if not os.path.exists("data/emails.csv"):
        print("Error: emails.csv not found in data/ directory. Please ensure the file exists.")
        return
    
    try:
        df = pd.read_csv("data/emails.csv")
        print("Dataset loaded successfully. Sample rows:")
        print(df.head())
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Map categories to numerical labels
    category_map = {"Incident": 0, "Request": 1, "Problem": 2, "Change": 3}
    if not all(df['type'].isin(category_map.keys())):
        print("Error: Invalid categories in emails.csv. Expected: Incident, Request, Problem, Change. Found:")
        print(df['type'].unique())
        return
    df['label'] = df['type'].map(category_map)
    print("Categories mapped to labels successfully.")

    # Split dataset
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['email'].values, df['label'].values, test_size=0.2, random_state=42
    )
    print(f"Dataset split: {len(train_texts)} training samples, {len(val_texts)} validation samples.")

    # Initialize tokenizer and model
    try:
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels=4  # Updated to 4 for Incident, Request, Problem, Change
        )
        print("BERT model and tokenizer initialized successfully.")
    except Exception as e:
        print(f"Error initializing BERT model/tokenizer: {e}")
        return

    # Create datasets
    train_dataset = EmailDataset(train_texts, train_labels, tokenizer)
    val_dataset = EmailDataset(val_texts, val_labels, tokenizer)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./model",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # Train the model
    try:
        trainer.train()
        print("Model training completed.")
    except Exception as e:
        print(f"Error during training: {e}")
        return

    # Save model and tokenizer with explicit path check
    model_dir = "./model"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    try:
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)
        print(f"Model and tokenizer saved to {model_dir}")
        # Verify files
        expected_files = ["pytorch_model.bin", "config.json", "vocab.txt", "tokenizer.json", 
                         "special_tokens_map.json", "tokenizer_config.json"]
        saved_files = os.listdir(model_dir)
        print(f"Files in {model_dir}: {saved_files}")
        if not all(f in saved_files for f in expected_files):
            print("Warning: Some expected files are missing from the model directory.")
    except Exception as e:
        print(f"Error saving model/tokenizer: {e}")
        return

# Classification function
def classify_email(text: str):
    print("Starting email classification...")
    try:
        # Check if model directory exists and contains required files
        model_dir = "./model"
        required_files = ["pytorch_model.bin", "config.json", "vocab.txt", "tokenizer.json", 
                         "special_tokens_map.json", "tokenizer_config.json"]
        
        if not os.path.exists(model_dir) or not all(os.path.exists(os.path.join(model_dir, f)) for f in required_files):
            print(f"Error: Model directory {model_dir} is incomplete or missing. Required files: {required_files}")
            print("Attempting fallback to pre-trained model...")
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            model = BertForSequenceClassification.from_pretrained(
                "bert-base-uncased",
                num_labels=4  # Updated to 4 for Incident, Request, Problem, Change
            )
            print("Fallback pre-trained model loaded. Note: This model is not fine-tuned for your categories.")
        else:
            try:
                tokenizer = BertTokenizer.from_pretrained(model_dir, use_fast=False)
                model = BertForSequenceClassification.from_pretrained(model_dir)
                print(f"Fine-tuned model and tokenizer loaded successfully from {model_dir}")
            except Exception as e:
                print(f"Error loading tokenizer/model from {model_dir}: {e}")
                print("Attempting fallback to pre-trained model...")
                tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
                model = BertForSequenceClassification.from_pretrained(
                    "bert-base-uncased",
                    num_labels=4  # Updated to 4 for Incident, Request, Problem, Change
                )
                print("Fallback pre-trained model loaded. Note: This model is not fine-tuned for your categories.")

        # Tokenize input
        inputs = tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        print("Input tokenized successfully. Input shape:")
        print(f"Input IDs: {inputs['input_ids'].shape}, Attention Mask: {inputs['attention_mask'].shape}")

        # Perform inference
        with torch.no_grad():
            outputs = model(**inputs)
            print("Inference completed. Output logits shape:")
            print(outputs.logits.shape)

        # Process output
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_label = torch.argmax(probs, dim=-1).item()
        print(f"Predicted label index: {predicted_label}, Probabilities: {probs.tolist()}")

        categories = ["Incident", "Request", "Problem", "Change"]  # Updated to include Change
        if 0 <= predicted_label < len(categories):
            print(f"Category assigned: {categories[predicted_label]}")
            return categories[predicted_label]
        else:
            print("Predicted label out of range. Returning Uncategorized.")
            return "Uncategorized"

    except Exception as e:
        print(f"Classification error: {e}")
        return "Uncategorized"








