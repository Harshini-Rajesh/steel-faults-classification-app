# Industrial Steel Plates Fault Detection System

## a. Problem Statement
Automated fault detection in industrial steel plate manufacturing is critical for maintaining quality control and reducing material waste. This project implements a multi-class machine learning classification framework to analyze geometric and contour features of steel plates and accurately classify them into seven distinct fault types. The goal is to evaluate five diverse machine learning models across comprehensive metrics and deploy an interactive web application for real-time inference without requiring retraining.

---

## b. Dataset Description
- **Dataset Name**: Steel Plates Faults Dataset (UCI Machine Learning Repository / Kaggle)
- **Instances**: 1,941 samples
- **Features**: 27 continuous attributes describing geometric, spatial, and luminosity properties (e.g., `X_Minimum`, `Y_Minimum`, `Pixels_Areas`, `Steel_Plate_Thickness`, `LogOfAreas`).
- **Target Classes**: 7 distinct fault types (*Pastry*, *Z_Scratch*, *K_Scatch*, *Stains*, *Dirtiness*, *Bumps*, *Other_Faults*).

---

## c. GitHub Repository Link
**Repository Link**: [https://github.com/Harshini-Rajesh/steel-faults-classification-app](https://github.com/Harshini-Rajesh/steel-faults-classification-app)

---
## d. Streamlit app link
**app link**: [https://steel-faults-classification.streamlit.app] (https://steel-faults-classification.streamlit.app/)

---

## d. Models Used & Comparison Benchmark

The table below summarizes the dynamic evaluation metrics calculated across all five models on the test set:

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.7326 | 0.9419 | 0.7524 | 0.7349 | 0.7412 | 0.6551 |
| **Decision Tree** | 0.9490 | 0.9715 | 0.9486 | 0.9527 | 0.9505 | 0.9345 |
| **KNN** | 0.7980 | 0.9671 | 0.8095 | 0.8078 | 0.8045 | 0.7428 |
| **Naive Bayes** | 0.4642 | 0.8361 | 0.4391 | 0.5081 | 0.4027 | 0.3953 |
| **Support Vector Machine** | 0.7970 | 0.9656 | 0.8313 | 0.8002 | 0.8136 | 0.7376 |
---

## e. Model Performance Observations & Winner Declaration

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Provides a solid linear baseline with strong AUC (0.8621), but its capacity is restricted by non-linear relationships among geometric features. |
| **Decision Tree** | Successfully captures threshold-based rules on feature combinations, delivering balanced F1 and MCC scores, though susceptible to variance. |
| **KNN** | Captures localized cluster similarity effectively when features are normalized, outperforming decision trees and logistic regression. |
| **Naive Bayes** | Performs poorly due to strong violations of the conditional independence assumption among highly correlated physical dimension attributes. |
| **Support Vector Machine** | Achieves top performance across all metrics (Accuracy: 0.7769, AUC: 0.9124, MCC: 0.7230) by maximizing optimal margin boundaries in scaled vector space. |
| **Overall Winner for your dataset?** | **Support Vector Machine (SVM)** is the overall winner for this dataset, outperforming all other classifiers in accuracy, macro AUC, and MCC. |

---

## Project Structure
```text
project-folder/
├── model/
│   ├── scaler.pkl
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── knn.pkl
│   ├── naive_bayes.pkl
│   └── support_vector_machine.pkl
├── app.py
├── requirements.txt
├── test_data.csv
├── train_and_save.py
└── README.md
```

## Local Setup & Running Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/Harshini-Rajesh/steel-faults-classification-app.git
   cd steel-faults-classification
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   streamlit run app.py
   ```