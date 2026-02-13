Fraud Detection – Model Comparison 
This project is part of Elevate Labs Task‑14: Model Comparison & Best Model Selection using a synthetic credit card fraud detection dataset.

1. Objective
Build a fraud detection model on a transactional dataset.

Compare multiple ML algorithms on the same train–test split.

Evaluate models with accuracy, precision, recall and F1‑score.

Detect overfitting and select the best model based on business needs (fraud detection).

2. Dataset
File: synthetic_fraud_dataset.csv

Columns (examples):

amount: Transaction amount.

transaction_type: ATM / POS / Online / QR, etc.

merchant_category: Food, Travel, Clothing, Electronics, Grocery etc.

country: Country of transaction.

hour: Hour of the day (0–23).

device_risk_score, ip_risk_score: Risk scores between 0 and 1.

is_fraud: Target label (1 = fraud, 0 = genuine).

ID columns transaction_id and user_id are dropped because they do not generalize.

3. Preprocessing
Dropped: transaction_id, user_id.

Separated features and target: X (all columns except is_fraud), y = is_fraud.

Train–test split: 80% train, 20% test with stratify=y to keep fraud ratio same in both sets.

Numeric features: amount, hour, device_risk_score, ip_risk_score scaled with StandardScaler.

Categorical features: transaction_type, merchant_category, country encoded with OneHotEncoder.

Preprocessing done via a single ColumnTransformer so all models share the same transformed features.

4. Models
Compared models (all wrapped in a Pipeline(preprocessor + model)):

Logistic Regression (LogisticRegression, max_iter=1000, class_weight="balanced").

Decision Tree (DecisionTreeClassifier, class_weight="balanced").

Random Forest (RandomForestClassifier, class_weight="balanced").

Support Vector Machine (SVC, class_weight="balanced", probability=True).

Note: In one run, all four models achieved perfect scores (1.0) on both train and test sets for accuracy, precision, recall and F1‑score.
This suggests the synthetic dataset is relatively easy and the patterns are very clear, so all models can separate fraud vs non‑fraud perfectly on this split.

Metric table
Example output (model_comparison_fraud.csv):

Model	Accuracy	Precision	Recall	F1_score	Train_Accuracy
LogisticRegression	1.0	1.0	1.0	1.0	1.0
DecisionTree	1.0	1.0	1.0	1.0	1.0
RandomForest	1.0	1.0	1.0	1.0	1.0
SVM	1.0	1.0	1.0	1.0	1.0
The bar chart model_performance_fraud.png shows all bars at 1.0 for every model and metric.

5. Model generalization & overfitting
Train accuracy and test accuracy are both 1.0 for all four models.

Because there is no gap between train and test scores, there is no sign of overfitting on this dataset/split.

To stress‑test generalization further, we could use cross‑validation or different random splits, but for this task one split is sufficient.

6. Best model selection
Fraud detection is usually imbalanced and high‑risk, so we care most about:

High recall: catch as many fraudulent transactions as possible.

Good precision: avoid too many false alarms.

Stable generalization.

Since all models achieved perfect metrics, the tie‑breaker is:

Simplicity and interpretability → Logistic Regression.

Inference speed and robustness → Logistic Regression or Random Forest.

For this task, I selected Logistic Regression as the best model because:

It reaches 1.0 on all evaluation metrics on both train and test.

It is simple, fast, and easier to interpret in a real banking environment.

The final saved model is stored as:

best_fraud_model.joblib – a Pipeline that includes preprocessing and the Logistic Regression classifier.

7. Files in this repository
synthetic_fraud_dataset.csv – dataset used.

model_comparison.ipynb – Jupyter notebook with full EDA, training, and evaluation.

model_comparison_fraud.csv – comparison table with metrics for all models.

model_performance_fraud.png – bar chart comparing Accuracy, Precision, Recall, F1.

best_fraud_model.joblib – saved best model (preprocessing + Logistic Regression).

AIML_task_14.pdf – task description (from Elevate Labs).

8. How to run
git clone <https://github.com/PranithaBokketi/task14-model-comparison>
cd task14-fraud-detection


pip install -r requirements.txt

# Run the notebook or script
jupyter notebook model_comparison.ipynb
# or
python src/task14_model_comparison.py
