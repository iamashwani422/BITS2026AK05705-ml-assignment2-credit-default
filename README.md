# Credit Card Default Prediction

**ML Assignment 2 · M.Tech (AIML), BITS WILP**
**Student:** Ashwani · **BITS ID:** 2026AK05705

**Live app:** https://credit-default-bits2026ak05705.streamlit.app/

---

## a. Problem statement

A credit card issuer wants to know, in advance, which customers are likely to **default on their payment next month**, so it can act early (reduce limits, follow up, offer restructuring) instead of discovering the loss afterwards.

This is framed as a **binary classification** problem: given a customer's credit limit, demographics, and their last six months of repayment status, bill amounts and payments, predict whether they will default (`Y = 1`) or not (`Y = 0`).

Five classifiers are trained and compared on the same data, and the best one is identified. An interactive Streamlit app loads the saved models and evaluates them on uploaded test data.

---

## b. Dataset description

| Item | Detail |
|---|---|
| Name | Default of Credit Card Clients |
| Source | UCI Machine Learning Repository (Yeh & Lien, 2009) – Taiwan, Apr–Sep 2005 |
| Instances | 30,000 |
| Features | 23 (all numeric) |
| Target | `Y` – default payment next month (1 = yes, 0 = no) |
| Missing values | None |
| Class balance | 77.9 % no default · 22.1 % default (imbalanced) |

**Features**

| Column | Meaning |
|---|---|
| X1 | Credit limit (NT$) |
| X2 | Sex (1 = male, 2 = female) |
| X3 | Education (1 = graduate school, 2 = university, 3 = high school, 4 = other) |
| X4 | Marital status (1 = married, 2 = single, 3 = other) |
| X5 | Age (years) |
| X6 – X11 | Repayment status, Sep → Apr (-2/-1 = paid duly, 1–9 = months of delay) |
| X12 – X17 | Bill statement amount, Sep → Apr (NT$) |
| X18 – X23 | Amount actually paid, Sep → Apr (NT$) |

**Preprocessing**

- Dropped the stored row-index column (`Unnamed: 0`).
- Stratified 80 / 20 train–test split (`random_state = 42`) → 24,000 train / 6,000 test rows. The test split is saved as `test_data.csv`.
- `StandardScaler` applied inside each model pipeline, so the saved `.joblib` file contains both the scaler and the classifier.
- `class_weight="balanced"` used for Logistic Regression, Decision Tree and SVM to counter the class imbalance.
- SVM (RBF kernel) trained on a stratified 8,000-row sample of the training set to keep training time practical.

---

## c. GitHub repository

**Link:** https://github.com/iamashwani422/BITS2026AK05705-ml-assignment2-credit-default.git

```
├── app.py                  # Streamlit app
├── train_models.ipynb      # training + evaluation notebook
├── requirements.txt
├── README.md
├── credit_card.csv         # full dataset
├── test_data.csv           # held-out test split (features + Y)
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── support_vector_machine.joblib
    └── metrics.csv
```

**Run locally**

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute train_models.ipynb   # optional: retrain
streamlit run app.py
```

---

## d. Models used

Logistic Regression, Decision Tree (max_depth = 6, min_samples_leaf = 50), K-Nearest Neighbors (k = 15, distance-weighted), Gaussian Naive Bayes, and Support Vector Machine (RBF kernel).

### Comparison table (test set, 6,000 rows)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.6797 | 0.7081 | 0.3672 | 0.6202 | 0.4613 | 0.2718 |
| Decision Tree | 0.7812 | **0.7651** | 0.5049 | 0.5403 | **0.5220** | **0.3807** |
| KNN | **0.8077** | 0.7303 | **0.6183** | 0.3406 | 0.4393 | 0.3564 |
| Naive Bayes | 0.7525 | 0.7249 | 0.4515 | 0.5539 | 0.4975 | 0.3386 |
| Support Vector Machine | 0.7768 | 0.7487 | 0.4959 | 0.5509 | **0.5220** | 0.3778 |

> Because 78 % of customers do not default, a model that always predicts "no default" would already score 78 % accuracy. **AUC, F1 and MCC** are therefore the more meaningful metrics for this dataset.

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Highest recall (0.62) – it catches the most defaulters – but the lowest precision (0.37), accuracy and MCC. With balanced class weights it flags many customers as risky, producing a lot of false alarms. Its linear decision boundary cannot capture threshold effects such as "two or more months late", which limits it. |
| Decision Tree | Best AUC (0.765) and MCC (0.381), and joint-best F1. The tree relies mainly on the most recent repayment status (X6 accounts for ~71 % of feature importance), which is the single strongest predictor of default. Limiting depth to 6 and leaf size to 50 prevented overfitting, and the model stays easy to interpret. |
| KNN | Highest accuracy (0.81) and precision (0.62), but the lowest recall (0.34) – it misses about two-thirds of defaulters. KNN has no class-weight option, so the majority "no default" class dominates most neighbourhoods. A good example of why accuracy is misleading on imbalanced data. |
| Naive Bayes | Middle of the table on every metric. Its independence assumption is clearly violated: the six monthly bill amounts (X12–X17) are strongly correlated with each other (r ≥ 0.80), so the same evidence is counted several times. The amounts are also highly skewed, which does not fit the Gaussian assumption. Still, it is fast and gives reasonable recall (0.55). |
| Support Vector Machine | A close second: ties the Decision Tree on F1 (0.522) with nearly the same MCC (0.378) and the second-best AUC (0.749). The RBF kernel captures non-linear patterns well. Downside: by far the slowest to train, so it was trained on an 8,000-row sample – with the full training set it may close the gap further. |
| **Overall winner** | **Decision Tree** – best AUC and MCC, joint-best F1, fast to train and predict, and interpretable, which matters for explaining credit decisions. SVM is a close alternative. |

---

## Streamlit app features

- Upload a test CSV (or use the bundled `test_data.csv`)
- Model selection dropdown for all five models
- All six evaluation metrics for the selected model
- Confusion matrix and classification report
- "Compare all models" tab with the full metrics table and a bar chart
- Download predictions as CSV
