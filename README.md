# Loan Approval Prediction System

A Data Science mini-project that predicts whether a bank loan application will be **Approved** or **Rejected**, based on applicant financial and demographic data.

**Domains covered:** DS (Financial Analytics) · ML (Approval Prediction) · AI (Credit Evaluation)

---

## 📌 Overview

Manual loan approval is slow, inconsistent, and prone to human bias. This project builds an end-to-end pipeline — from raw applicant data to a trained, deployable model — that automates loan approval decisions using machine learning.

The pipeline covers data cleaning, exploratory data analysis (EDA), feature engineering, model training and comparison, and evaluation, culminating in a saved model ready for inference on new applicants.

---

## 📊 Dataset

- **Source:** `loan_approval_dataset.csv`
- **Size:** 4,269 records × 13 features
- **Target:** `loan_status` (Approved / Rejected) — 62% / 38% split
- **No missing values**; 28 records with invalid negative asset values were corrected during cleaning

**Feature groups:**

| Category | Fields |
|---|---|
| Demographics | `no_of_dependents`, `education`, `self_employed` |
| Financials | `income_annum`, `loan_amount`, `loan_term` |
| Credit | `cibil_score` (300–900) |
| Assets | `residential_assets_value`, `commercial_assets_value`, `luxury_assets_value`, `bank_asset_value` |

---

## 🛠 Tools & Technologies

- **Language:** Python 3
- **Environment:** Jupyter Notebook / Google Colab
- **Data handling:** pandas, numpy
- **Visualization:** matplotlib, seaborn
- **Machine Learning:** scikit-learn (Logistic Regression, Decision Tree, Random Forest)
- **Model persistence:** joblib

---

## ⚙️ Project Workflow

```
Data Collection → Data Cleaning → EDA → Feature Engineering →
Model Building → Model Evaluation → Deployment
```

1. **Data Cleaning** — stripped whitespace from columns/text, verified no missing values, clipped 28 invalid negative asset values to 0
2. **EDA** — analyzed class balance, CIBIL score distribution, income vs. loan amount, correlations, and approval rate by demographics
3. **Feature Engineering** — created `total_assets_value`, `loan_to_income_ratio`, and `asset_to_loan_ratio`
4. **Model Building** — trained and compared Logistic Regression, Decision Tree, and Random Forest on an 80/20 stratified split
5. **Model Evaluation** — accuracy, precision, recall, F1-score, ROC-AUC, confusion matrix, feature importance

---

## 📈 Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.9157 | 0.9088 | 0.8638 | 0.8857 | 0.9745 |
| **Decision Tree** ⭐ | 0.9988 | 1.0000 | 0.9969 | **0.9984** | 0.9985 |
| Random Forest | 0.9977 | 1.0000 | 0.9938 | 0.9969 | 1.0000 |

**Best model:** Decision Tree (highest F1-score, near-perfect ROC-AUC)

### Key Insights
- **CIBIL score dominates** the decision (~79% feature importance) — approved applicants average ≈703, rejected average ≈429
- **Education and self-employment barely matter** (~62% approval rate either way) — approval is financially driven, not demographic
- **Non-linear boundary** — tree-based models outperform Logistic Regression, meaning approval isn't a simple linear rule
- **Engineered ratios help** — `asset_to_loan_ratio` outperforms any single raw asset column

---

## 📁 Repository Structure

```
├── data/
│   ├── loan_approval_dataset.csv       # raw dataset
│   ├── loan_featured.csv               # cleaned + feature-engineered dataset
│   ├── loan_train.csv                  # 80% train split
│   └── loan_test.csv                   # 20% test split
├── notebooks/
│   └── Loan_Approval_Prediction_Project.ipynb
├── models/
│   └── loan_approval_model.pkl         # trained Decision Tree model
├── report/
│   └── Loan_Approval_Prediction_Report.docx
├── presentation/
│   └── Loan_Approval_Prediction_System.pptx
└── README.md
```

---

## 🚀 Usage

```python
import joblib

# Load the trained model bundle
bundle = joblib.load('models/loan_approval_model.pkl')
model = bundle['model']

# new_applicant: a DataFrame row with the same engineered features
prediction = model.predict(new_applicant)
probability = model.predict_proba(new_applicant)
```

---

## ⚠️ Limitations

- The near-perfect model scores suggest the dataset was generated with a strong CIBIL-driven rule; real-world lending data will be noisier
- No temporal or macroeconomic context (interest rates, economic conditions)
- Single snapshot dataset — no repayment/default outcome tracked over time

## 🔭 Future Scope

- Validate on real, noisier bank lending data
- Add explainability (SHAP values) for regulatory transparency
- Deploy as a live API / web form for real-time credit evaluation
- Track repayment/default outcomes for long-term credit-risk modeling

---

## 📚 References

- [scikit-learn documentation](https://scikit-learn.org)
- [pandas documentation](https://pandas.pydata.org)
- [seaborn documentation](https://seaborn.pydata.org)
- [Matplotlib documentation](https://matplotlib.org)
