# Text model comparison

| Model | Accuracy | F1 Score (macro) | Training Time (s) |
|---|---|---|---|
| Linear SVM | 89.5% | 84.2% | 1.78 |
| Logistic Regression | 86.9% | 80.7% | 0.72 |
| Gradient Boosting | 86.0% | 82.8% | 43.96 |
| Random Forest | 85.8% | 81.3% | 5.67 |
| Naive Bayes | 71.8% | 48.6% | 0.01 |

Best test accuracy: **Linear SVM** (89.5%). Weighted F1 (accounts for class imbalance differently than macro F1) is also saved in the CSV alongside these columns.