# 🔍 Log Classification for Proactive Platform Monitoring

## 📘 Project Overview

This capstone project focuses on building a machine learning pipeline to classify system log messages into categories such as `error` and `info`. The goal is to proactively alert platform engineers about potential issues before they escalate, enabling faster response and improved system reliability.

---

## 🧠 Problem Statement

Modern platforms generate thousands of log messages daily. Manually inspecting these logs is inefficient and error-prone. This project automates log classification to identify critical messages (`error`) and distinguish them from routine ones (`info`), helping engineers detect brewing issues early.

---

## 📊 Exploratory Data Analysis (EDA)

### 📁 Dataset Overview
- **Rows**: 28,971
- **Columns**: 4
- **Fields**:
  - `Time`: timestamp of log entry
  - `service_name`: source of the log
  - `detected_level`: log severity (e.g., info, error)
  - `log`: unstructured log message text

### ✅ Missing Values
- No missing values detected across any columns.
- **Key Takeaway**: Dataset is complete and clean.

### 🔢 Data Types Distribution
- `object`: text-based columns
- `int64`: numeric identifiers and timestamps
- **Key Takeaway**: Majority of columns are text-based, followed by numeric types.

### 🧮 Service Names by Detected Level
- Grouped bar chart shows distribution of `info`, `error`, and `debug` logs across services.
- **Key Takeaway**: Error-level logs are concentrated in specific services—potential system hotspots.

### 🔤 Frequent Words in Logs
- Tokenized `log` column and calculated word frequencies.
- Visualized using bar chart and word cloud.
- **Key Takeaway**: Common terms include `timeout`, `connection`, `error`, suggesting recurring system issues.

### ⏰ Log Messages by Hour
- Converted timestamps from UTC to EST (UTC−5).
- Extracted hourly distribution of logs.
- **Key Takeaway**: Most logs are generated after business hours—likely due to ETL batch operations.

### 📏 Log Message Length & Frequency
- Most log messages are under 500 bytes.
- **Key Takeaway**: ERROR logs tend to be longer due to stack traces, while INFO logs are typically concise.

---

## 🧠 Feature Engineering & Clustering

### 📐 TF-IDF Embeddings
- Applied TF-IDF vectorization to extract semantic patterns.
- 2D projection shows consistent clustering between train and test sets.

### 🔍 Clustering Techniques
- Applied DBSCAN and KMeans to reduced embeddings.
- DBScan clustering did perform well on identifying Anomaly Detections when applied on larger datasets
- **Key Takeaway**: Both methods generalize well across splits, capturing consistent structure.

---

## ⚖️ Scaling & KDE Visualization

### ✅ Scaling Best Practices
- Label encoding applied only to target variable
- Train-test split performed before scaling
- Avoided scaling one-hot encoded categorical features

### 📈 Importance of Scaling
- **StandardScaler** centers SVD components to mean 0 and scales to unit variance.
- Applied only to continuous features (SVD components).
- Prevents data leakage and preserves categorical meaning.

### 📊 KDE Output Summary
- **Before Scaling**: SVD components show natural separation but varied ranges.
- **After Scaling**: Distributions are centered and standardized.
- **Key Takeaway**: KDE confirms scaling preserves class separation while normalizing feature behavior.

---

## 🔧 Pipeline Summary

### 1. Data Preparation
- Loaded logs from `Log-Classification_Capstone_Final.csv`
- Filtered for `error` and `info`
- Encoded target labels using `LabelEncoder`

### 2. Feature Engineering
- **Text**: TF-IDF vectorization
- **Time**: Extracted hour from timestamp
- **Service**: One-hot encoded `service_name`
- **Dimensionality Reduction**: Truncated SVD (3 components)
- **Scaling**: StandardScaler applied to SVD components only

### 3. Modeling
- Train/test split (80/20) with stratified sampling
- Combined scaled SVD + unscaled metadata
- Trained classifiers with `GridSearchCV`:
  - Logistic Regression
  - K-Nearest Neighbors
  - Decision Tree
  - Random Forest
  - Support Vector Machine (Linear Kernel)
- Included Dummy Classifier as baseline

### 4. Evaluation
- ✅ Cross-validated predictions (`cross_val_predict`)
- ✅ Confusion matrix visualization
- ✅ ROC-AUC curves for model discrimination
- ✅ Precision, recall, F1-score, and support metrics

---

# 📊 Baseline Model Evaluation

This section summarizes the performance of five machine learning classifiers trained to distinguish between `error` and `info` log messages. The evaluation was conducted on a held-out test set using precision, recall, and F1-score as key metrics.

---

## ✅ Model Performance Summary

| Model               | Precision | Recall | F1-Score | Support |
|--------------------|-----------|--------|----------|---------|
| Logistic Regression| 0.9993    | 0.9993 | 0.9993   | 5795    |
| K-Nearest Neighbors| 0.9998    | 0.9998 | 0.9998   | 5795    |
| Decision Tree      | 0.9991    | 0.9991 | 0.9991   | 5795    |
| Random Forest      | 0.9995    | 0.9995 | 0.9995   | 5795    |
| Support Vector Machine| 0.9995 | 0.9995 | 0.9995   | 5795    |

---

## 🧠 Key Takeaway

All models achieved near-perfect scores across all metrics, indicating extremely strong performance on the test set. This suggests that:

- The feature engineering pipeline (TF-IDF, SVD, metadata fusion) is highly effective.
- The classification task may be relatively easy given the current dataset.
- Adding more logs mainly **"info"** category made the dataset more **imbalanced** and the precision/recalls scores went down to 83%
  - Random Forest could handle class imbalances when combined with class weighting
  - Gradient Boosting models that handle Class Imbalances applying hyperparameter tuning using scale_pos_weight on minority subsets were identified but are not documented here

---

## ⚠️ Recommendation

To ensure these results generalize to real-world logs:

- Validate with cross-validation or a separate holdout set.
- Test on unseen or noisy logs from production environments.
- Monitor for alert fatigue if models over-predict `error`.


## Confusion Matrix & Metric Analysis

| Metric        | Description                                                                 |
|---------------|-----------------------------------------------------------------------------|
| **Accuracy**  | Overall correctness of predictions                                          |
| **Precision** | % of predicted `error` logs that are truly errors (controls false positives)|
| **Recall**    | % of actual `error` logs correctly identified (controls false negatives)    |
| **F1-score**  | Harmonic mean of precision and recall                                       |

### Why FP and FN Matter

- **False Positives (FP)**: Info logs misclassified as errors  
  → May cause alert fatigue or unnecessary investigations

- **False Negatives (FN)**: Error logs misclassified as info  
  → **Critical risk**: Engineers miss real issues, leading to downtime or failures

### Priority: **Recall > Precision > Accuracy**
In this use case, **recall is most important**. Missing an actual error (FN) is far more dangerous than over-alerting (FP). The system must prioritize catching every potential issue—even if it means a few false alarms.

---

## 🏁 Future Enhancements

This project demonstrates a scalable approach to log classification, enabling proactive platform monitoring. Future enhancements may include:

- Infuse more Log data with more unique and random cases, to identify the effectiveness of the models
- Infuse random error data to ensure the models identify the errors that are anomalies and not being trained upon
- Real-time log ingestion and classification
- Integration with alerting systems
- Deployment as a microservice for production use
