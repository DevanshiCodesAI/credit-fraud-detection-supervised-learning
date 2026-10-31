# Credit Card Fraud Detection — Summary Report

## Business problem

This project looks at how machine learning can help identify fraudulent credit card transactions. A payment company needs to catch fraud without blocking too many genuine customers or overwhelming its review team. The dataset is highly imbalanced: only around 0.17% of transactions are fraud. Because of that, accuracy is not a good measure on its own. A model that predicts every transaction as legitimate could still be about 99.83% accurate while failing to catch any fraud. I used precision, recall, F1-score and PR-AUC to compare models. Precision describes how many flagged transactions are actually fraud, recall describes how much of the fraud is caught, and PR-AUC summarizes the precision-recall tradeoff across thresholds.

## Data and method

The notebook uses a stratified sample of up to 50,000 rows, then divides it into training and test sets while preserving the fraud ratio. It creates `Amount_log` and `Hour` features and scales those features; the PCA-transformed V1–V28 columns are left as they are. The test set is kept untouched. On the training data, Logistic Regression is tested with class weighting, SMOTE and random undersampling. Random Forest and XGBoost are also trained. XGBoost uses `scale_pos_weight`, and a randomized search selects hyperparameters using average precision. The best Logistic Regression strategy, model scores and threshold results should be checked in the notebook output.

## Model recommendation

The notebook evaluates tuned XGBoost at the default threshold, the threshold that gives its highest F1-score, and a threshold selected to meet a recall target of at least 0.90 when possible. After running the notebook, enter the chosen model and threshold here: **XGBoost and 0.384**. At that threshold, test precision is **1.000**, recall is **0.882**, F1-score is **0.938**, and PR-AUC is **0.933**. A lower threshold can find more fraud but may also create more false alerts. The operating threshold should therefore be agreed with the risk team based on review capacity and the impact of blocking genuine purchases, rather than selected from a model score alone.

## Cost-benefit result

For the business simulation, the notebook assumes that each caught fraud saves ₹4,500 and each flagged transaction costs ₹150 to investigate. It reports true positives, false positives, missed fraud, money saved, review cost, money lost and net benefit for thresholds 0.1, 0.2, 0.3, 0.5 and the F1-best threshold. The best tested threshold by net benefit was **0.384**, with an estimated net benefit of **₹326250** on the test portion, or approximately **₹326250 per 50,000 transactions** after scaling. These are estimates based on the exam's simplified costs; they are not measured company savings. The business-best threshold may differ from the F1-best one because the metrics value errors differently.

## Next steps

Before using a model in production, I would evaluate it on newer transactions and monitor fraud rate, recall, PR-AUC, false-alert volume and customer complaints over time. I would also collect analyst decisions as feedback for future model updates. A real deployment should include secure real-time scoring, drift checks, an alert review process and a fallback plan if the model or data pipeline is unavailable. The cost assumptions and chosen threshold should be revisited as fraud patterns and business priorities change.

> **Note:** The CSV dataset was not attached with the exam PDF, so model metrics and rupee results are left as placeholders. Run the notebook with `creditcard.csv`, then replace the bracketed values using its outputs. The notebook also contains a cell that generates a report with the run's actual values as `summary_report.md`.
