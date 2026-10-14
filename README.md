# AI Customer Support Ticket Classifier

A local NLP classification lab for customer support tickets. This project provides an interactive Streamlit dashboard to train, evaluate, and compare different machine learning models for text classification. The initial CSV is a small starter dataset for learning and code verification, not a production-quality training corpus.

## Project Overview

This project enables you to:
- Train multiple classification models (Logistic Regression, Linear SVM, Neural Network)
- Experiment with different text feature extraction methods (TF-IDF, CountVectorizer)
- Compare model performance using accuracy, F1-score, and confusion matrices
- Inspect misclassified tickets and add corrected examples to improve the dataset
- Predict categories for new support tickets

## Project Structure

```
customer_ticket_classifier/
├── app.py                 # Streamlit dashboard interface
├── train_model.py         # Model training and evaluation logic
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── data/
│   └── tickets.csv       # Labeled training data
├── models/
│   └── ticket_classifier.joblib  # Saved trained pipeline
└── venv/                 # Virtual environment (not tracked in git)
```

## Features

### Supported Models
1. **Logistic Regression** - Fast, interpretable linear classifier
2. **Linear SVM** - Support Vector Machine for text classification
3. **Neural Network (MLP)** - Multi-layer perceptron with feature scaling

### Text Feature Options
- **TF-IDF** - Term frequency-inverse document frequency with sublinear scaling
- **Count** - Simple term frequency counts
- **N-grams** - Unigrams (1) or unigrams + bigrams (1-2)
- **Stop words** - Option to remove English stop words

### Dashboard Features
- **Experiment Settings**: Configure model, features, and hyperparameters
- **Dataset Overview**: View ticket count, categories, and examples per category
- **Performance Metrics**: Accuracy, macro F1-score, per-category precision/recall/F1
- **Confusion Matrix**: Visual representation of classification performance
- **Error Analysis**: View and analyze misclassified tickets
- **Experiment History**: Track and compare multiple experiments
- **Live Prediction**: Predict categories for new tickets
- **Dataset Expansion**: Add corrected labeled examples to improve the model

## Recent Updates

### Neural Network Stability Fix (October 2026)
Fixed NaN/Inf errors during neural network training by:
- Adding `StandardScaler` to normalize feature vectors in the pipeline
- Disabling early_stopping to prevent validation split issues with small datasets
- Increasing max_iter to 1000 for better convergence
- Using adaptive learning rate and conservative hyperparameters
- Smaller hidden layer (50 neurons) for stability

## 1. Requirements

- Python 3.10 or newer
- Windows, macOS, or Linux
- Internet is only needed to install the Python packages; model training runs locally.

## 2. Setup on Windows

Open Command Prompt or PowerShell in this folder:

```powershell
py -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Setup on macOS/Linux

Open Terminal in this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Run the Dashboard

```powershell
streamlit run app.py
```

Streamlit will print a local URL, usually `http://localhost:8501`. Open it in your browser.

## 5. Run Training from Terminal

```powershell
python train_model.py
```

This will:
- Load the dataset from `data/tickets.csv`
- Train a model with default settings (Logistic Regression + TF-IDF)
- Print accuracy and classification report
- Save the trained pipeline to `models/ticket_classifier.joblib`

## 6. Experiments to Try

1. **Model Comparison**: Logistic Regression vs Linear SVM vs Neural Network
2. **Feature Engineering**: TF-IDF vs CountVectorizer
3. **N-gram Analysis**: Unigrams vs unigrams + bigrams
4. **Stop Words**: Test with and without stop word removal
5. **Regularization**: Vary C parameter for Logistic Regression and SVM (0.1 to 5.0)

Change one setting at a time and compare accuracy and macro F1. Results can go down as well as up.

## 7. How the Code Works

### Data Loading (`load_data()`)
- Reads `data/tickets.csv`
- Removes rows with missing text or category
- Strips whitespace and filters empty strings
- Ensures each category has at least 2 examples

### Pipeline Creation (`make_pipeline()`)
Creates a Scikit-learn pipeline with:
1. **Vectorizer**: Converts text to numerical features (TF-IDF or Count)
2. **Scaler** (Neural Network only): StandardScaler normalizes features
3. **Classifier**: The chosen model (Logistic Regression, SVM, or MLP)

### Training Process (`run_experiment()`)
1. Loads and validates the dataset
2. Creates stratified train/test split (75/25)
3. Builds the pipeline with selected parameters
4. Fits the pipeline on training data
5. Predicts on test data
6. Calculates metrics: accuracy, classification report, confusion matrix
7. Returns results including the trained pipeline

### Model-Specific Configurations

**Logistic Regression**
- Uses regularization parameter C (inverse of regularization strength)
- Default: 500 max iterations
- Class weight: None (balanced by stratified split)

**Linear SVM**
- Uses regularization parameter C
- Default: 5000 max iterations
- Suitable for high-dimensional text data

**Neural Network (MLPClassifier)**
- Hidden layer: 50 neurons
- Alpha (L2 regularization): 0.0001
- Max iterations: 1000
- Adaptive learning rate starting at 0.001
- Includes StandardScaler for feature normalization
- Early stopping disabled for stability with small datasets

## 8. Dataset Format

The training data should be a CSV file with two columns:

```csv
text,category
"I was charged twice for my monthly subscription",billing
"Please refund the payment I made yesterday",billing
"My invoice shows the wrong amount",billing
"I can't login to my account",technical
"The app crashes when I try to upload",technical
```

- **text**: The customer support ticket content
- **category**: The target label/class for classification

## 9. Adding New Examples

You can add new labeled examples through the Streamlit dashboard:
1. Navigate to "Add a corrected labeled example" section
2. Enter the ticket text
3. Select the correct category
4. Click "Save example to dataset"
5. The example is appended to `data/tickets.csv`
6. Run a new experiment to retrain with the expanded dataset

## 10. Important Evaluation Notes

### Dataset Size
The starter dataset is small and manually written, so accuracy numbers are only a demonstration that the workflow runs. A larger real-world dataset with diverse, human-labeled tickets is needed for meaningful model comparison.

### Data Leakage
Keep a final test set untouched when you begin iteratively adding examples; otherwise test results can become overly optimistic. The current implementation uses a random stratified split, so adding examples will change the test set composition.

### Neural Network Training
Neural networks require more data to perform well. With small datasets, they may underperform compared to simpler models like Logistic Regression. The current configuration is tuned for stability with limited data.

## 11. Troubleshooting

### Neural Network NaN/Inf Error
If you encounter NaN/Inf errors during neural network training:
- Ensure the dataset has enough examples per category (at least 10-20 recommended)
- Try reducing the model complexity (fewer neurons)
- Increase the regularization (alpha parameter)
- The current implementation includes fixes for this issue

### Memory Issues
Large datasets with many features may cause memory issues:
- Use CountVectorizer instead of TF-IDF (simpler features)
- Reduce ngram range to unigrams only
- Limit max_features in the vectorizer

### Poor Performance
- Ensure categories are balanced (similar number of examples per class)
- Add more diverse examples to the dataset
- Try different preprocessing (stop words, n-grams)
- Consider data augmentation techniques

## 12. Future Improvements

- Add cross-validation for more robust evaluation
- Implement grid search for hyperparameter tuning
- Add more model options (Random Forest, XGBoost, BERT-based models)
- Include data augmentation for small datasets
- Add model explainability features (SHAP values, feature importance)
- Implement model versioning and experiment tracking
- Add API endpoint for programmatic predictions

## License

This project is provided as-is for educational and development purposes.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.
