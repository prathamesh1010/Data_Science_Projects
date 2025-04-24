
Email Classification & PII Masking API
An intelligent, privacy-conscious solution for automated email classification tailored for customer support teams. This system categorizes emails into four types (incident, request, change, problem) and redacts Personally Identifiable Information (PII) (e.g., email addresses, phone numbers) using rule-based methods, ensuring data privacy while streamlining ticket triaging.
Features

Email Classification: Uses a fine-tuned BERT model to classify emails into operational categories.
PII Masking: Redacts sensitive information like emails, phone numbers, and SSNs using regex-based rules.
Interactive Interface: Built with Streamlit for easy email input and result visualization.
Deployment Ready: Supports local deployment and Hugging Face Spaces.

Use Case
Customer support teams receive hundreds of emails daily. Manual sorting is time-consuming and error-prone. This project automates email categorization and PII redaction, enabling faster routing to the appropriate teams while ensuring compliance with data privacy standards.
Tech Stack



Component
Technology Used



Model
BERT (Hugging Face Transformers)


Data Preprocessing
Python, Regex


Frontend/Backend
Streamlit


Deployment
Local, Hugging Face Spaces, Docker (optional)


Project Structure
project/
├── app.py                  # Streamlit app for email classification
├── model.py                # Model loading and inference
├── train_model.py          # Model training script
├── utils.py                # Utility functions (e.g., PII masking)
├── requirements.txt        # Project dependencies
├── data/
│   └── emails.csv          # Dataset with 'email' and 'type' columns
├── model/
│   ├── config.json         # Model configuration
│   ├── pytorch_model.bin   # Model weights
│   ├── vocab.txt           # Tokenizer vocabulary
│   ├── tokenizer.json      # Tokenizer configuration
│   ├── tokenizer_config.json
│   ├── special_tokens_map.json
│   └── labels.txt          # Category labels

Installation

Clone the Repository:
git clone <repo_url>
cd project/


Set Up a Virtual Environment (recommended):
python -m venv env
source env/bin/activate  # On Windows: .\env\Scripts\activate


Install Dependencies:
pip install -r requirements.txt



Usage

(Optional) Train the Model:

Ensure data/emails.csv contains labeled data (email, type columns).
Run:python train_model.py


This saves the fine-tuned model and tokenizer to model/.


Run the Streamlit App:
streamlit run app.py


Access the app at http://localhost:8501.
Enter an email, click "Classify" to view the predicted category, confidence, and PII-masked email.



Example
Input Email:
Please reset my password for john.doe@example.com ASAP!

Output (in Streamlit):

Masked Email: Please reset my password for <EMAIL> ASAP!
Predicted Category: request
Confidence: 97%

Model Details

Base Model: bert-base-uncased
Categories: incident, request, change, problem
Training Metrics (approximate, post-fine-tuning):
Accuracy: ~90%
F1-Score: ~89%
Loss: ~0.3


Class Imbalance: Handled via weighted loss during training.

PII Masking
Sensitive information is masked using regex rules. Examples:

john.doe@example.com → <EMAIL>
+91-9876543210 → <PHONE>
123-45-6789 → <SSN>

Deployment

Local: Run streamlit run app.py and access at http://localhost:8501.
Hugging Face Spaces: Upload app.py, model/, and requirements.txt; configure as a Streamlit app.
Docker (optional): Containerize for cloud deployment.

Troubleshooting

PyTorch Error: If you see Tried to instantiate class '__path__._path'..., reinstall PyTorch:pip install torch --force-reinstall


Model/Tokenizer Error: If Error loading tokenizer/model from ./model: 'added_tokens' appears, ensure model/ contains all required files. Re-run train_model.py or manually download bert-base-uncased:from transformers import AutoModelForSequenceClassification, AutoTokenizer
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=4)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model.save_pretrained("./model")
tokenizer.save_pretrained("./model")


Update Dependencies: Ensure huggingface_hub is up-to-date:pip install huggingface_hub --upgrade



Future Enhancements

Add visualizations (e.g., confidence score charts) in Streamlit.
Support multilingual email classification.
Implement active learning for continuous model improvement.
Export classification logs for analytics dashboards.
Expand PII masking to include addresses and IDs.

License
This project is licensed under the MIT License.
Contributing
Contributions are welcome! Please open a pull request or issue on GitHub.
Contact

Email: your.email@example.com
GitHub: YourGitHub

