# Credit Card Fraud Detection

End-to-end ML pipeline for credit card fraud detection on the [Kaggle Credit Card Fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (284,807 transactions, 0.17% fraud).

## TL;DR

| Model | Recall | Precision | F2 | Realistic Cost |
|:---|:---|:---|:---|:---|
| **RF + class_weight='balanced' @ 0.3** | **0.85** | **0.92** | **0.861** | **\$16,400** ⭐ |
| XGBoost @ 0.3 | 0.86 | 0.86 | 0.857 | \$16,870 |
| RF + SMOTE @ 0.3 | 0.89 | 0.70 | 0.843 | \$18,400 |
| RF (F2-tuned via GridSearchCV) @ 0.3 | 0.87 | 0.79 | 0.852 | \$17,400 |

**Production recommendation:** RF + class_weight='balanced' + threshold=0.3.
Best F2 (the fraud-appropriate metric), lowest realistic cost when accounting for false-positive customer churn risk, and the simplest model.

## Why this project

Fraud detection is a textbook imbalanced classification problem with asymmetric costs:
- Missed fraud (FN): ~\$1,000+ direct loss
- False alarm (FP): ~\$5 direct + ~\$200 hidden (trust erosion, churn risk)

This project explores the trade-offs between recall, precision, and realistic business cost — rather than chasing a single metric.

## Methods

### Notebook 1: Baseline (`01_baseline.ipynb`)
- EDA: class distribution, feature correlations
- Logistic Regression (with and without `class_weight='balanced'`)
- RandomForest with threshold tuning (0.5 → 0.3)

### Notebook 2: Advanced (`02_advanced.ipynb`)
- ROC-AUC and Precision-Recall curves
- XGBoost with `scale_pos_weight`
- F1 vs F2 metric comparison (F2 prioritizes recall, fitting fraud's cost asymmetry)
- SMOTE vs `class_weight='balanced'`
- GridSearchCV (F2-optimized hyperparameter search)
- Feature importance analysis (RF vs XGBoost)
- Cost simulation with hidden customer-churn costs

## Key insights

1. **AUC and F1 alone are misleading on imbalanced data.** F2 score and direct cost simulation drive better decisions than F1 maximization.
2. **Hyperparameter tuning didn't beat the default RF.** GridSearchCV's best params (CV F2 = 0.805) underperformed the original RF on test (F2 = 0.861). "Simpler is better" confirmed.
3. **SMOTE wins on direct cost but loses on realistic cost.** Synthetic oversampling improves recall but inflates false positives, flipping the result once customer-churn cost is modeled.
4. **V14 dominates feature importance in both RF and XGBoost.** XGBoost concentrates 60% on V14 alone, making it more fragile than RF for production.
5. **Threshold meanings differ across models.** RF's 0.3 ≠ XGBoost's 0.3 — probability distributions are calibrated differently.

## Reproduce

```bash
git clone https://github.com/aeril716/fraud-detection.git
cd fraud-detection

# Install dependencies (uv required: brew install uv)
uv sync

# Download dataset from Kaggle and place at:
# data/creditcard.csv

# Launch Jupyter
uv run jupyter lab
```

## Project structure
fraud-detection/
├── notebooks/
│   ├── 01_baseline.ipynb    # EDA + LR + RF baseline
│   └── 02_advanced.ipynb    # ROC/PR, XGBoost, SMOTE, GridSearch, Feature importance
├── pyproject.toml           # Dependencies (uv)
├── uv.lock                  # Locked versions
└── .python-version          # Python 3.13

## Tech stack

Python 3.13 · scikit-learn · XGBoost · imbalanced-learn (SMOTE) · pandas · matplotlib · seaborn · Jupyter · uv