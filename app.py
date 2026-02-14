import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(page_title="ML Model Deployment", layout="wide")

st.title("Machine Learning Model Deployment App")
st.write("Upload test dataset, select a model, and evaluate performance.")

# --------------------------------------------------
# Model Selection
# --------------------------------------------------
st.sidebar.header("Model Selection")

model_choice = st.sidebar.selectbox(
    "Choose a Model",
    ["Logistic Regression", "KNN","Naive Bayes","Random Forest","Decision Tree","XG Boost"]
)

# Load model based on selection
def load_model(choice):
    if choice == "Logistic Regression":
        return joblib.load("Logistic_Regression.pkl")
    elif choice == "KNN":
        return joblib.load("KNN.pkl")
    elif choice == "Naive Bayes":
        return joblib.load("Naive_Bayes.pkl")
    elif choice == "Random Forest":
        return joblib.load("Random_Forest.pkl")
    elif choice == "Decision Tree":
        return joblib.load("Decision_Tree.pkl")
    elif choice == "XG Boost":
        return joblib.load("XG_Boost.pkl")

model = load_model(model_choice)

st.sidebar.success(f"{model_choice} Loaded Successfully")

# --------------------------------------------------
# File Upload
# --------------------------------------------------
uploaded_file = st.file_uploader("Upload Test Dataset (CSV only)", type=["csv"])

if uploaded_file is not None:

    try:
      df = pd.read_csv(uploaded_file)
    except UnicodeDecodeError:
      df = pd.read_csv(uploaded_file, encoding="latin1")
    st.subheader("Uploaded Data Preview")
    st.dataframe(df.head())

    # --------------------------------------------------
    # Target Column Selection
    # --------------------------------------------------
    target_column = st.selectbox("Select Target Column", df.columns)

    if st.button("Run Prediction"):

        try:
            X = df.drop(columns=[target_column])
            y = df[target_column]

            # Prediction
            y_pred = model.predict(X)

            # --------------------------------------------------
            # Metrics
            # --------------------------------------------------
            st.subheader("Evaluation Metrics")

            col1, col2, col3, col4 = st.columns(4)

            col1.metric("Accuracy", f"{accuracy_score(y, y_pred):.4f}")
            col2.metric("Precision", f"{precision_score(y, y_pred, average='weighted'):.4f}")
            col3.metric("Recall", f"{recall_score(y, y_pred, average='weighted'):.4f}")
            col4.metric("F1 Score", f"{f1_score(y, y_pred, average='weighted'):.4f}")

            # --------------------------------------------------
            # Confusion Matrix
            # --------------------------------------------------
            st.subheader("Confusion Matrix")

            cm = confusion_matrix(y, y_pred)

            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)

            # --------------------------------------------------
            # Classification Report
            # --------------------------------------------------
            st.subheader("Classification Report")

            report = classification_report(y, y_pred)
            st.text(report)

        except Exception as e:
            st.error(f"Error during prediction: {e}")

else:
    st.info("Please upload a CSV file to begin.")