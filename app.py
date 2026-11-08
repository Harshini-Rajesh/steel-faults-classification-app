import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef, confusion_matrix
)

st.set_page_config(
    page_title="Steel Faults Classification Dashboard",
    layout="wide"
)

st.markdown("""
    <style>
    /* Center-aligned container for main title and subtitle in Arial font */
    .header-container {
        text-align: center;
        font-family: Arial, Helvetica, sans-serif;
        margin-bottom: 2rem;
    }
    .main-title {
        font-size: 32px; 
        font-weight: 700; 
        margin-bottom: 4px;
    }
    .sub-title {
        font-size: 16px; 
        color: #555555; 
        margin-top: 0px;
    }
    
    /* Subheaders sizing */
    h2, h3 {
        font-family: Arial, Helvetica, sans-serif;
    }
    
    /* Dynamic model name colour using accent blue */
    .highlight-model {
        color: #2563eb;
        font-weight: 700;
    }
    </style>
""", unsafe_allow_html=True)

# Center-aligned Header and Subheading
st.markdown("""
    <div class="header-container">
        <div class="main-title">Industrial Steel Plates Fault Detection System</div>
        <div class="sub-title">Machine Learning Assignment Evaluation App</div>
    </div>
""", unsafe_allow_html=True)

st.markdown("---")

@st.cache_resource
def load_artifacts():
    try:
        scaler = joblib.load('model/scaler.pkl')
        models = {
            "Logistic Regression": joblib.load('model/logistic_regression.pkl'),
            "Decision Tree": joblib.load('model/decision_tree.pkl'),
            "KNN": joblib.load('model/knn.pkl'),
            "Naive Bayes": joblib.load('model/naive_bayes.pkl'),
            "Support Vector Machine": joblib.load('model/support_vector_machine.pkl')
        }
        return scaler, models, None
    except Exception as e:
        return None, None, str(e)

scaler, models, err = load_artifacts()

if err:
    st.error(f"Failed to load model artifacts: {err}. Check the 'model/' directory.")
    st.stop()

st.sidebar.header("Control Panel")
model_choice = st.sidebar.selectbox("Select Classification Model", list(models.keys()))

st.sidebar.markdown("---")
st.sidebar.subheader("Dataset Management")
uploaded_file = st.sidebar.file_uploader("Upload Test Data (CSV)", type=["csv"])

if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)
    st.sidebar.success("Custom test CSV loaded.")
else:
    if os.path.exists("test_data.csv"):
        test_df = pd.read_csv("test_data.csv")
    else:
        st.error("test_data.csv not found locally or uploaded.")
        st.stop()

feature_cols = test_df.columns[:27].tolist()
target_cols = ['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']

X_test = test_df[feature_cols]
y_true_onehot = test_df[target_cols]
y_test = y_true_onehot.idxmax(axis=1).astype('category').cat.codes
class_names = y_true_onehot.columns.tolist()

selected_model = models[model_choice]

if model_choice in ["Logistic Regression", "KNN", "Support Vector Machine"]:
    X_test_processed = scaler.transform(X_test)
else:
    X_test_processed = X_test

preds = selected_model.predict(X_test_processed)
try:
    probs = selected_model.predict_proba(X_test_processed)
except Exception:
    probs = np.zeros((len(preds), len(class_names)))

acc = accuracy_score(y_test, preds)
try:
    auc = roc_auc_score(y_test, probs, multi_class='ovr', average='macro')
except Exception:
    auc = 0.0
prec = precision_score(y_test, preds, average='macro', zero_division=0)
rec = recall_score(y_test, preds, average='macro', zero_division=0)
f1 = f1_score(y_test, preds, average='macro', zero_division=0)
mcc = matthews_corrcoef(y_test, preds)

st.markdown(f"<h3>Performance Evaluation: <span class='highlight-model'>{model_choice}</span></h3>", unsafe_allow_html=True)

col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Accuracy", f"{acc:.4f}")
col2.metric("AUC Score", f"{auc:.4f}")
col3.metric("Precision", f"{prec:.4f}")
col4.metric("Recall", f"{rec:.4f}")
col5.metric("F1 Score", f"{f1:.4f}")
col6.metric("MCC Score", f"{mcc:.4f}")

st.markdown("---")

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots(figsize=(6, 4))
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax)
    plt.xlabel("Predicted Class", fontsize=9)
    plt.ylabel("Actual Class", fontsize=9)
    st.pyplot(fig)

with col_right:
    st.subheader("Sample Predictions Preview")
    preview_df = X_test.head(5).copy()
    preview_df['Actual Fault'] = y_true_onehot.idxmax(axis=1).head(5).values
    preview_df['Predicted Fault'] = pd.Series(preds).map(dict(enumerate(class_names))).head(5).values
    st.dataframe(preview_df[['Actual Fault', 'Predicted Fault']], use_container_width=True)

st.markdown("---")
st.subheader("Overall Model Comparison")

@st.cache_data
def compute_all_models_metrics():
    summary_results = []
    for name, mdl in models.items():
        if name in ["Logistic Regression", "KNN", "Support Vector Machine"]:
            X_proc = scaler.transform(X_test)
        else:
            X_proc = X_test
            
        p = mdl.predict(X_proc)
        try:
            prob = mdl.predict_proba(X_proc)
        except Exception:
            prob = np.zeros((len(p), len(class_names)))
            
        ac = accuracy_score(y_test, p)
        try:
            au = roc_auc_score(y_test, prob, multi_class='ovr', average='macro')
        except Exception:
            au = 0.0
        pr = precision_score(y_test, p, average='macro', zero_division=0)
        rc = recall_score(y_test, p, average='macro', zero_division=0)
        f1_val = f1_score(y_test, p, average='macro', zero_division=0)
        mc = matthews_corrcoef(y_test, p)
        
        summary_results.append({
            "ML Model Name": name,
            "Accuracy": round(ac, 4),
            "AUC": round(au, 4),
            "Precision": round(pr, 4),
            "Recall": round(rc, 4),
            "F1": round(f1_val, 4),
            "MCC": round(mc, 4)
        })
    return pd.DataFrame(summary_results)

comparison_df = compute_all_models_metrics()
st.dataframe(comparison_df, use_container_width=True, hide_index=True)

st.markdown("---")
st.subheader("Model Performance Observations")

st.markdown("""
- **Logistic Regression**: 
  - **Key Metrics**: Accuracy: 73.26% | AUC: 0.9419 | MCC: 0.6551
  - **Analysis**: Serves as a reliable linear baseline, achieving a strong macro-AUC of 0.9419. However, its overall accuracy is capped around 73.26% with an MCC of 0.6551. Because the steel plates dataset involves complex, non-linear geometric and contour boundaries across multiple defect classes, a purely linear decision boundary struggles to cleanly separate overlapping feature distributions.

- **Decision Tree Classifier**: 
  - **Key Metrics**: Accuracy: 94.90% | F1: 0.9505 | MCC: 0.9345
  - **Analysis**: Emerges as the clear top-performing model across every single metric, yielding a remarkable 94.90% accuracy, 95.05% F1-score, and an outstanding MCC of 0.9345. Tree-based axis-aligned splits map naturally onto physical steel plate attributes (such as perimeter, length, and conveyor anomalies), capturing intricate threshold rules effectively while maintaining robust generalization on unseen test instances.

- **K-Nearest Neighbors (KNN)**: 
  - **Key Metrics**: Accuracy: 79.80% | AUC: 0.9671
  - **Analysis**: Performs very competitively with an accuracy of 79.80% and a high AUC of 0.9671. By relying on local neighborhood proximity in the feature space, it handles multi-class clusters effectively; however, its performance remains sensitive to how well the attributes were preprocessed and scaled during training.

- **Gaussian Naive Bayes**: 
  - **Key Metrics**: Accuracy: 46.42% | F1: 0.4027
  - **Analysis**: Exhibits the lowest performance across the board, recording an accuracy of 46.42% and an F1-score of 0.4027. This drop is expected because Naive Bayes assumes strict conditional independence between features—an assumption that is heavily violated by the steel dataset, where geometric and contour measurements are inherently correlated.

- **Support Vector Machine (SVM)**: 
  - **Key Metrics**: Accuracy: 79.70% | AUC: 0.9656 | F1: 0.8136 | MCC: 0.7376
  - **Analysis**: Demonstrates strong, stable results (Accuracy: 79.70%, AUC: 0.9656, F1: 0.8136, MCC: 0.7376). By projecting features into a higher-dimensional space and optimizing hyperplane margins, the SVM effectively handles multi-class boundaries, sitting just behind the Decision Tree as one of the most reliable models for this problem.
""")