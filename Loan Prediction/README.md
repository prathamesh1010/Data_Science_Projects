Loan Prediction using PySpark

This project predicts loan eligibility based on applicant data using PySpark for distributed data processing and machine learning.

Features
Applicant Income: Income of the loan applicant.
Coapplicant Income: Income of the co-applicant.
Loan Amount: Requested loan amount.
Loan Amount Term: Duration of the loan (in months).
Credit History: Whether the applicant has a good credit history.
Dependents, Gender, Education, Self-Employed, Property Area: Additional features.
Loan Status (Target): Whether the loan was approved (Yes/No).
Workflow
Data Loading:

Load the dataset into PySpark DataFrame.
Data Preprocessing:

Handle missing values and perform feature encoding.
Normalize numerical features.
Modeling:

Train ML models using PySpark MLlib (e.g., Logistic Regression, Random Forest).
Split data into train/test sets and evaluate models.
Prediction:

Results
The model predicts loan approval with high accuracy, providing insights for financial institutions to assess loan eligibility.


