# 🕵️ Fraud Detection Prediction App

A machine learning project that detects potentially fraudulent financial transactions in real time, built on a large-scale synthetic mobile money transaction dataset (6.3M+ records). The project covers exploratory data analysis on highly imbalanced transaction data, feature engineering, a scikit-learn preprocessing + classification pipeline, and deployment as an interactive Streamlit web application.

![Python](https://img.shields.io/badge/Python-3.14-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Pipeline-orange?logo=scikitlearn)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## 📌 Overview

Fraud detection is a high-stakes, highly imbalanced classification problem in banking and fintech — fraudulent transactions typically make up a tiny fraction of total volume, yet cost institutions heavily when missed. This project builds a classification pipeline on **6.36 million** simulated mobile money transactions to flag a transaction as **fraudulent** or **legitimate** in real time, and wraps the model in a Streamlit interface where a transaction's details can be entered to get an instant prediction.

**Live demo:** _add your deployed Streamlit Cloud / HuggingFace Spaces link here_

---

## 🗂️ Dataset

**Source:** [Fraud Detection Dataset — Kaggle](https://www.kaggle.com/datasets/amanalisiddiqui/fraud-detection-dataset?resource=download)

A synthetic dataset (based on the PaySim simulator) modeling mobile money transactions, generated from real transaction logs from an African mobile money service, injected with fraudulent behavior for research purposes.

- **6,362,620** transactions across a simulated 30-day period (744 time steps)
- **8,213** confirmed fraudulent transactions (**0.13%** of all transactions — a severe class imbalance)
- **16** transactions independently flagged by the platform's own rule-based system (`isFlaggedFraud`), a tiny fraction of actual frauds — showing how much fraud simple rule-based flags miss

| Attribute | Description |
|---|---|
| step | Time unit (1 step = 1 hour of simulated time) |
| type | Transaction type: PAYMENT, TRANSFER, CASH_OUT, CASH_IN, DEBIT |
| amount | Transaction amount |
| nameOrig | Customer ID initiating the transaction |
| oldbalanceOrg / newbalanceOrig | Sender's account balance before / after the transaction |
| nameDest | Recipient customer/merchant ID |
| oldbalanceDest / newbalanceDest | Recipient's account balance before / after the transaction |
| isFraud | Target — 1 if the transaction is fraudulent |
| isFlaggedFraud | Whether the platform's rule engine flagged the transaction |

---

## 🔍 Approach

### 1. Exploratory Data Analysis
- Confirmed extreme class imbalance (0.13% fraud rate) and zero missing values across all 6.3M rows.
- Broke down fraud rate by transaction `type` and found fraud is **exclusively** concentrated in `TRANSFER` (0.77% fraud rate) and `CASH_OUT` (0.18% fraud rate) transactions — `PAYMENT`, `CASH_IN`, and `DEBIT` transactions had zero recorded fraud.
- Visualized the (highly right-skewed) transaction amount distribution on a log scale, and compared amount distributions between fraudulent and legitimate transactions.
- Plotted fraud incidents over time (by `step`) to check for temporal patterns.
- Reviewed a correlation matrix across balance and amount fields to understand relationships between account balances and transaction amount.

**Key insights from EDA:**
- Fraud only occurs within `TRANSFER` and `CASH_OUT` transaction types — this alone is a powerful filtering signal.
- A large share of transactions (~1.19M) show the sender's balance dropping to exactly zero after a `TRANSFER`/`CASH_OUT` — a classic fraud pattern (account draining) worth engineering into a feature.
- `newbalanceDest` shows the strongest correlation with `amount` (0.46) among the numeric features, while origin balances are near-perfectly correlated with each other (0.9988), as expected.

### 2. Feature Engineering
- Engineered `balanceDiffOrig` (sender's balance change) and `balanceDiffDest` (recipient's balance change) to explicitly capture the balance-draining pattern often seen in fraud.
- Dropped high-cardinality identifier columns (`nameOrig`, `nameDest`) and the rule-based `isFlaggedFraud` column (to avoid leaking a weak existing signal) before modeling.
- Dropped `step` after using it for the temporal EDA, since it doesn't generalize as a predictive feature.

### 3. Modeling Pipeline
- Built a single **scikit-learn `Pipeline`** combining preprocessing and modeling so the exact same transformations are applied at both training and inference time:
  - `ColumnTransformer` → `OneHotEncoder` on the categorical `type` feature
  - `StandardScaler` on numeric features (`amount`, `oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest`)
  - `LogisticRegression` as the classifier
- Split the data 70/30 with stratification on `isFraud` to preserve the (very small) fraud class ratio in both train and test sets.
- Serialized the entire fitted pipeline with `joblib` as a single deployable artifact (`fraud_detection_pipeline.pkl`), so the web app doesn't need to replicate any preprocessing logic manually.

### 4. Deployment
- Built an interactive **Streamlit** web app (`fraud_detection.py`) where transaction details (type, amount, sender/receiver balances) are entered through form inputs, and the pipeline returns an instant fraud / not-fraud prediction.

---

## 🛠️ Tech Stack

- **Language:** Python
- **Data Analysis:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** scikit-learn (Pipeline, ColumnTransformer, OneHotEncoder, StandardScaler, LogisticRegression)
- **Model Persistence:** joblib
- **Web App / Deployment:** Streamlit

---

## 📁 Project Structure

```
Fraud-Detection-Prediction/
│
├── fraud_detection.py                # Streamlit web application
├── analysis_model.ipynb              # EDA, feature engineering, and pipeline training notebook
├── AIML_Dataset.csv                  # Raw transaction dataset (PaySim-based)
├── fraud_detection_pipeline.pkl      # Trained preprocessing + Logistic Regression pipeline
├── requirements.txt                  # Project dependencies
└── README.md
```

> **Note:** The raw dataset (~490 MB) is not committed to this repository due to GitHub file size limits. Download it directly from the [Kaggle link above](https://www.kaggle.com/datasets/amanalisiddiqui/fraud-detection-dataset?resource=download) and place it in the project root before running the notebook.

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/AnsalCD/fraud-detection-prediction.git
cd fraud-detection-prediction
```

### 2. Set up a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate      # On Windows: .venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
streamlit run fraud_detection.py
```
The app will open automatically in your browser at `http://localhost:8501`.

---

## 🖥️ App Preview

_Add a screenshot or GIF of the running Streamlit app here, e.g.:_
```markdown
![App Screenshot](assets/app_screenshot.png)
```

---

## 📈 Future Improvements

- Apply class-imbalance techniques (SMOTE, class weighting, or undersampling) and compare against tree-based models (Random Forest, XGBoost) — Logistic Regression alone is a strong, interpretable baseline but may under-recall fraud cases at this imbalance ratio.
- Report and track precision/recall/F1 and PR-AUC (more informative than accuracy on a 0.13%-positive-class problem) alongside a confusion matrix for the final chosen model.
- Add the engineered `balanceDiffOrig` / `balanceDiffDest` features into the modeling pipeline explicitly, since they were engineered but not yet fed into the final feature set.
- Add SHAP-based explainability to the app so flagged transactions come with a "why" for a fraud analyst.
- Deploy to Streamlit Community Cloud / HuggingFace Spaces for public access and link it here.

---

## 🙏 Acknowledgements

- Dataset: [Fraud Detection Dataset — Kaggle (amanalisiddiqui)](https://www.kaggle.com/datasets/amanalisiddiqui/fraud-detection-dataset?resource=download), based on the PaySim synthetic mobile money simulator.

---

## 👤 Author

**Ansal Caitan D'Souza**
Data & Business Analytics Professional | BFSI Domain | Cloud & AI-augmented Analytics

- GitHub: [AnsalCD](https://github.com/AnsalCD)
- Portfolio: [datascienceportfol.io/ansalcd](https://datascienceportfol.io/ansalcd)
- Tableau Public: [public.tableau.com/app/profile/ansalcd](https://public.tableau.com/app/profile/ansalcd)

---

*If you found this project useful, consider giving it a ⭐ on GitHub!*
