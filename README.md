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

## Key Observations and Performance Analysis

## Model Performance Observations

| ML Model Name | Primary Metrics | Key Technical Observations & Theoretical Justification |
| :--- | :--- | :--- |
| **Logistic Regression** | **Accuracy:** 73.26%<br>**AUC:** 0.9419<br>**MCC:** 0.6551 | Serves as a reliable linear baseline, achieving a strong macro-AUC of 0.9419. However, its overall accuracy is capped around 73.26% with an MCC of 0.6551. Because the steel plates dataset involves complex, non-linear geometric and contour boundaries across multiple defect classes, a purely linear decision boundary struggles to cleanly separate overlapping feature distributions. |
| **Decision Tree** | **Accuracy:** 94.90%<br>**F1-Score:** 0.9505<br>**MCC:** 0.9345 | Emerges as the clear top-performing model across every single metric, yielding a remarkable 94.90% accuracy, 95.05% F1-score, and an outstanding MCC of 0.9345. Tree-based axis-aligned splits map naturally onto physical steel plate attributes (such as perimeter, length, and conveyor anomalies), capturing intricate threshold rules effectively while maintaining robust generalization on unseen test instances. |
| **KNN** | **Accuracy:** 79.80%<br>**AUC:** 0.9671 | Performs very competitively with an accuracy of 79.80% and a high AUC of 0.9671. By relying on local neighborhood proximity in the feature space, it handles multi-class clusters effectively; however, its performance remains sensitive to how well the attributes were preprocessed and scaled during training. |
| **Naive Bayes** | **Accuracy:** 46.42%<br>**F1-Score:** 0.4027 | Exhibits the lowest performance across the board, recording an accuracy of 46.42% and an F1-score of 0.4027. This drop is expected because Naive Bayes assumes strict conditional independence between features—an assumption that is heavily violated by the steel dataset, where geometric and contour measurements are inherently correlated. |
| **Support Vector Machine** | **Accuracy:** 79.70%<br>**AUC:** 0.9656<br>**F1-Score:** 0.8136<br>**MCC:** 0.7376 | Demonstrates strong, stable results (Accuracy: 79.70%, AUC: 0.9656, F1: 0.8136, MCC: 0.7376). By projecting features into a higher-dimensional space and optimizing hyperplane margins, the SVM effectively handles multi-class boundaries, sitting just behind the Decision Tree as one of the most reliable models for this problem. |

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