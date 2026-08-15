# Text model comparison

| Model | Accuracy | F1 Score (macro) | Training Time (s) |
|---|---|---|---|
| Linear SVM | 89.2% | 83.9% | 2.03 |
| Logistic Regression | 86.9% | 80.6% | 1.12 |
| Gradient Boosting | 86.0% | 83.1% | 48.34 |
| Random Forest | 85.9% | 81.3% | 6.37 |
| Naive Bayes | 71.5% | 48.2% | 0.02 |

Best test accuracy: **Linear SVM** (89.2%). Weighted F1 (accounts for class imbalance differently than macro F1) is also saved in the CSV alongside these columns.