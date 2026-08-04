# AI/Machine-Learning-fraud-detection-model

***This project looks at a Fraud Detection System I developed for mobile money transactions, by training a Machine Learning model for AI applications that can detect instances of fraud. By analysing over 6.3 million transactions in a 30-day window using Python, it compares the detection capabilities of a Logistic Regression model against an XGBoost model, with test predictions that can flag transactions back to the original accounts involved for audit reviews.***

## Background 
Financial fraud is a growing threat to the global economy. Recent studies reveal that losses worldwide hit an estimated $442 billion in 2025 according to INTERPOL, while Nasdaq Verafin recorded $579.4 billion lost to bank fraud and scams that same year, up 9.2% from 2023.

The main challenges organisations face in combating financial fraud are that:
1. Fraud alerts in existing systems have become too high for manual review. Alert queues pile up to the tens of thousands per week at many institutions, with noone to effectively review or close each case,
2. Fraudsters are increasingly hard to distinguish from genuine customers. 1st party fraud now makes up 36% of all global fraud cases, more than doubling in a single year, and 
3. Criminals themselves are now using AI. INTERPOL noted that AI-enhanced fraud is 4.5 times more profitable than traditional methods, and agentic AI systems are able to autonomously plan and run entire fraud campaigns. 

Luckily, AI has helped address these by giving anti-fraud teams the same firepower. For instance, machine learning models are able to flag suspicious behaviour patterns and account takeovers in real time instead of relying on static rules. Graph analysis maps out fraud rings and mule networks that would look invisible transaction by transaction. Lastly, AI driven identity checks can catch potential fraudsters actors during onboarding processes before they capitalise.

This project aims to contribute to this field by developing a Python script that compares the performance of two machine learning models that learn to distinguish fraudulent transactions from legitimate ones, then exports a traceable, row-level report so predictions can be audited back to the original transaction.

## Data Overview

[PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1) is a synthetic dataset that simulates mobile money transactions. It was generated from real transaction logs from a mobile money service in an African country, then used to build a simulator that reproduces similar statistical patterns without exposing real customer data.

- **6,362,620** transactions across 30 days of simulated activity
- **8,213** fraudulent transactions  **(Fraud rate: 0.1291%) **
- Fraud occurs exclusively in two transaction types: **CASH_OUT** (4,116 cases) and **TRANSFER** (4,097 cases)

| Transaction Type | Count |
|---|---|
| CASH_OUT | 2,237,500 |
| PAYMENT | 2,151,495 |
| CASH_IN | 1,399,284 |
| TRANSFER | 532,909 |
| DEBIT | 41,432 |

## Methodology
The script runs through 4 key Steps:
1. **Data Ingestion + Inspection** : the script profiles the raw data, its transaction type distribution and fraud rate.
2. **Data transformation** : it then adds one-hot encode transaction type, and engineer two balance-discrepancy features (`errorBalanceOrig`, `errorBalanceDest`), that measure how much a transaction's before/after balances deviate from what they should be. PaySim's balance fields may not always update consistently, and the size of that inconsistency might turn out to be one of the strongest fraud signals in the dataset.
3. **Model Development & Evaluation** : the script then checks the performance of a Logistic Regression baseline and an XGBoost classifier, using a stratified train/test split to preserve the fraud ratio in both sets, and analyse their precision, recall, f1-score and support. 
4. **Export traceable predictions** : Every test transaction, its true label, both models' predictions, and account identifiers will then be saved to `test_predictions_traceable.csv`, so any flagged row can be traced back to a specific account and transaction, rather than trusted as a black-box score.

The reason behind choosing the two models outlined is that:
- **Logistic Regression** is a simple, transparent baseline that is easy to explain and has a useful floor, meaning that if a more complex model can't beat it, the added complexity is worthless.
- The open-source **XGBoost** library can capture non-linear interactions between features (e.g. a specific combination of amount, balance discrepancy, and transaction type) that a linear model can't. Comparing the two shows exactly how much performance that added complexity buys.

## Findings
The overall script generated the following:

Test set: 1,272,524 transactions (of which 1,643 were actual fraud cases)

| Metric | Meaning | Logistic Regression (Fraud) | XGBoost (Fraud) |
|---|---|---|---|
| Precision | Percentage of transactions flagged as Fraud that are true | 0.02 | 0.996 |
| Recall | Percentage of actual fraudulent transactions that were found by the model | 0.96 | 0.997 |
| F1-score | Evaluates the performance of classification models | 0.047 | 0.996 |
| Support | Number of actual fraud transactions in the test set. | 1,643 | 1,643 |

**Confusion matrices**

Logistic Regression:
```
[[1206847   64034]
 [     64    1579]]
```
XGBoost:
```
[[1270874       7]
 [      5    1638]]
```

**Takeaway:** 
The Logistic Regression model has a high recall rate (96%), but because of its low precision (2%), it will drown the system in false alarms. 64,034 legitimate transactions were flagged as fraud. Although precision isn't necessarily a must-have for fraud detection models unlike the recall rate, it is much more ideal for false positives (legitimate transactions labelled as fraud) to be as low as possible for those using the application, as the system will not overwhelm the users with false alarms they have to review. This is also reflected in its F1-score (4.7%), meaning that this model is low performing due to its bad precision. 

Meanwhile, XGBoost catches slightly more fraud cases (1,638 vs. 1,579 of 1,643 cases) while producing only 7 false positives. This is supported by its F1-score (99.6%), signifying its status as a well-performing and ideal fraud detector model.

## Traceable Predictions

`test_predictions_traceable.csv` contains every test transaction with:
- The original feature values
- Account identifiers (`nameOrig`, `nameDest`)
- The true label (`isFraud_actual`)
- Both models' predictions (`isFraud_pred_logreg`, `isFraud_pred_xgb`)

This turns model output into an auditable format. Any flagged transaction can be traced back to the accounts involved, rather than trusted purely on a probability score.

## How to Run Code

```bash
pip install pandas numpy scikit-learn xgboost pyarrow
python Paysim_fraud_model.py
```

Update `DATA_FILE` in the script to point to your local copy of the PaySim CSV (download from Kaggle — link above).

## Recommendations 

1. Tune the classification threshold using a precision-recall curve instead of the default 0.5 cutoff
2. Add cross-validation to confirm results are stable across folds
3. Plot XGBoost feature importances to understand which signals drive its predictions
4. Test on a held-out time window rather than a random split, to better simulate real-world deployment
