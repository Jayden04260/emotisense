# Audio model comparison

| Model | Accuracy | F1 Score (macro) | Training Time (s) |
|---|---|---|---|
| Random Forest | 52.8% | 50.8% | 7.23 |
| Gradient Boosting | 51.4% | 49.9% | 97.07 |
| SVM (baseline) | 49.7% | 48.6% | 39.62 |

Best test accuracy: **Random Forest** (52.8%). All models were trained and evaluated on the same actor-grouped ~80/20 split (StratifiedGroupKFold, random_state=42 - no speaker appears in both train and test, see 'Fairness & Generalisation Audit' in the README) over the same MFCC + ZCR + Chroma + RMS features, so accuracy differences come from the classifier, not the data split.