# QR Code Fraud Detection

## Overview
The **QR Code Fraud Detection** project is a machine learning-based application designed to distinguish between original ("First_Print") and counterfeit ("Second_Print") QR code images. The project consists of two main components:
1. **Exploratory Data Analysis (EDA)**: Extracts features from QR code images and trains a logistic regression model to classify them, with a confusion matrix visualization for evaluation.
2. **Model Building and Deployment**: Builds a convolutional neural network (CNN) using TensorFlow/Keras for classification and deploys it via a Flask web interface for real-time QR code authentication.

Key features include:
- **Feature Extraction**: Extracts six image features (mean intensity, contrast, sharpness, median intensity, edge density, and texture variation) for EDA.
- **Machine Learning**: Uses logistic regression for initial analysis and a CNN for robust classification.
- **Visualization**: Generates a confusion matrix to evaluate the logistic regression model.
- **Web Interface**: Provides a Flask-based interface for uploading and verifying QR code images.
- **Evaluation Metrics**: Computes accuracy, precision, recall, and F1-score for model performance.

## Prerequisites
To run this project, ensure you have the following:
- **Python**: Version 3.8 or higher
- **pip**: Python package manager
- **Jupyter Notebook**: For running `EDA.ipynb` and `Model_Building.ipynb`
- **Compatible Web Browser**: For accessing the Flask web interface (e.g., Chrome, Firefox)
- **Dataset**: A folder structure with QR code images (see Dataset Structure below)

## Dataset Structure
The dataset should be organized as follows:
```
QR_Code_Fraud_Detection/
├── First_Print/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── Second_Print/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
```
- **Path**: Place the dataset in `C:\Projects\Data_Science\QR_Code_Fraud_Detection` or update the `dataset_path` variable in both notebooks to match your directory.
- **Classes**: `First_Print` (original QR codes) and `Second_Print` (counterfeit QR codes).
- **Image Format**: Images should be in a standard format (e.g., JPG, PNG).

**Note**: The provided code indicates an issue where no images were found in the dataset directory (`Found 0 images belonging to 2 classes`). Ensure the dataset path is correct and contains images in the specified subfolders.

## Installation
1. **Clone the Repository** (if applicable):
   ```bash
   git clone <repository-url>
   cd qr-code-fraud-detection
   ```

2. **Create a Virtual Environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   Install the required Python packages using the provided `requirements.txt` or manually:
   ```bash
   pip install numpy opencv-python matplotlib scikit-learn tensorflow flask werkzeug pillow nest-asyncio
   ```

   Alternatively, save the following to `requirements.txt`:
   ```
   numpy==1.26.4
   opencv-python==4.10.0
   matplotlib==3.9.2
   scikit-learn==1.5.1
   tensorflow==2.17.0
   flask==3.0.3
   werkzeug==3.0.4
   pillow==10.4.0
   nest-asyncio==1.6.0
   ```

4. **Save the Notebooks**:
   - Save `EDA.ipynb` and `Model_Building.ipynb` in your project directory.
   - Alternatively, extract the code from the notebooks into Python scripts (`eda.py` and `model_building.py`) for command-line execution.

## Usage
### 1. Exploratory Data Analysis (`EDA.ipynb`)
This notebook performs feature extraction and trains a logistic regression model.

1. **Prepare the Dataset**:
   - Ensure the dataset is in the correct folder structure at `C:\Projects\Data_Science\QR_Code_Fraud_Detection`.
   - Update `DATA_PATH` in `EDA.ipynb` if the dataset is located elsewhere.

2. **Run the Notebook**:
   - Open `EDA.ipynb` in Jupyter Notebook:
     ```bash
     jupyter notebook EDA.ipynb
     ```
   - Execute all cells to:
     - Load and preprocess up to 100 images per class (`First_Print` and `Second_Print`).
     - Extract six features (mean intensity, contrast, sharpness, median intensity, edge density, texture variation).
     - Train a logistic regression model.
     - Generate and display a confusion matrix.

3. **Output**:
   - A confusion matrix plot showing the model's performance on the test set.
   - The notebook assumes grayscale images and resizes them to 300x300 pixels.

**Troubleshooting**:
- If no images are processed, verify the dataset path and ensure images exist in `First_Print` and `Second_Print` folders.
- Check for valid image formats and sufficient disk space.

### 2. Model Building and Deployment (`Model_Building.ipynb`)
This notebook builds a CNN model and deploys it via a Flask web interface.

1. **Prepare the Dataset**:
   - Same as for EDA; ensure the dataset path is correct.

2. **Run the Notebook**:
   - Open `Model_Building.ipynb` in Jupyter Notebook:
     ```bash
     jupyter notebook Model_Building.ipynb
     ```
   - Execute the cells up to the model training section:
     - Set up data generators for training and validation (80-20 split).
     - Define a CNN with three convolutional layers, max-pooling, and dense layers.
     - Attempt to train the model (note: the provided code shows a `ValueError: Must provide at least one structure` due to an empty dataset).

3. **Fixing the Training Error**:
   - The error occurs because the data generators found 0 images. Ensure the dataset path (`C:\Projects\Data_Science\QR_Code_Fraud_Detection`) contains images in `first_print` and `second_print` subfolders (case-sensitive).
   - Update the `dataset_path` if necessary and verify folder names match exactly (`first_print` vs. `First_Print`).

4. **Web Interface**:
   - After training, the notebook launches a Flask app for real-time QR code verification.
   - Access the interface at `http://localhost:3000` (a clickable link is displayed in the notebook).
   - Upload a QR code image to classify it as "Original" or "Counterfeit".

5. **Output**:
   - Training metrics (accuracy, loss) for each epoch (if training succeeds).
   - Validation metrics (accuracy, precision, recall, F1-score) after training.
   - A web interface for uploading and classifying QR code images.

**Troubleshooting**:
- **Empty Dataset**: Verify the dataset path and folder names. Ensure images are present and readable.
- **Flask Port Conflict**: If port 3000 is in use, change `app.run(port=3000)` to another port (e.g., `port=5000`).
- **Model Not Trained**: If training fails, the Flask app will fail to make predictions. Fix the dataset issue first.

## Features
- **EDA**:
  - Feature extraction using OpenCV and scikit-image (mean intensity, contrast, sharpness, median intensity, edge density, texture variation).
  - Logistic regression for initial classification.
  - Confusion matrix visualization using Matplotlib.
- **Model Building**:
  - CNN model with three convolutional layers (32, 64, 128 filters), max-pooling, and dense layers.
  - Data augmentation via `ImageDataGenerator` (rescaling).
  - Binary classification (original vs. counterfeit).
- **Deployment**:
  - Flask-based web interface for uploading and classifying QR code images.
  - Real-time predictions with results displayed as "Original" or "Counterfeit".

## Limitations
- **Dataset Dependency**: Both notebooks require a properly structured dataset. The provided code indicates no images were found, suggesting a path or naming issue.
- **Model Complexity**: The CNN is relatively simple and may not capture complex patterns in large datasets. Consider adding dropout or batch normalization for better generalization.
- **Grayscale vs. RGB**: EDA uses grayscale images, while the CNN expects RGB images, which may require preprocessing consistency.
- **Web Interface**: The Flask app is basic and lacks error handling for invalid uploads or large files.
- **Scalability**: The logistic regression model processes only 100 images per class, and the CNN training may fail with an empty dataset.

## Troubleshooting
- **Dataset Issues**:
  - Verify the dataset path and folder structure.
  - Ensure `first_print` and `second_print` folders contain valid images (case-sensitive naming).
  - Check image formats (JPG, PNG) and file permissions.
- **Training Error** (`ValueError: Must provide at least one structure`):
  - Ensure the dataset contains images in the expected subfolders.
  - Update `dataset_path` in `Model_Building.ipynb` to match the actual directory.
- **Flask Issues**:
  - Ensure `Uploads` folder exists in the project directory or create it manually.
  - Check for port conflicts and change the port if needed.
- **Dependency Errors**:
  - Verify all packages are installed (`pip list`).
  - Ensure compatible versions (e.g., TensorFlow 2.17.0 requires Python <= 3.12).

## Contributing
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature-name`).
3. Commit changes (`git commit -m "Add feature"`).
4. Push to the branch (`git push origin feature-name`).
5. Open a pull request.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contact
For questions or feedback, please open an issue on the repository.

---
*Generated on May 21, 2025*