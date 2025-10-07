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

## 🏁 Final Thoughts

This project demonstrates a scalable approach to log classification, enabling proactive platform monitoring. Future enhancements may include:

- Real-time log ingestion and classification
- Integration with alerting systems
- Deployment as a microservice for production use
