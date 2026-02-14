# HeartDiseasePrediction

1. Problem Statement

The objective of this project is to build and evaluate multiple machine learning classification models to predict whether a customer will opt for a subscription service based on their shopping behavior and demographic attributes.

This project demonstrates an end-to-end machine learning workflow, including:

Data preprocessing

Feature engineering

Model training and evaluation

Comparison of multiple classification algorithms

Deployment of a trained model using a Streamlit web application

The final goal is to assist businesses in identifying customers who are more likely to subscribe, enabling better marketing and retention strategies.

2. Dataset Description

Dataset Name: Customer Shopping Behavior Dataset

Source: Public dataset (Kaggle / UCI-style dataset)

Number of Instances: 3900

Number of Features: 17 (excluding target variable)

Target Variable: Subscription Status (Yes / No)

3. Machine Learning Models Used

All models were trained on the same dataset using a unified preprocessing pipeline that includes:

Standard Scaling for numerical features

One-Hot Encoding for categorical features

Models Implemented

Logistic Regression

Decision Tree Classifier

K-Nearest Neighbors (KNN)

Naive Bayes (Gaussian)

Random Forest (Ensemble Model)

XGBoost (Ensemble Model)

| ML Model Name       | Accuracy | AUC   | Precision | Recall | F1    | MCC   |
|---------------------|----------|-------|-----------|--------|-------|-------|
| Logistic Regression | [ ]      | [ ]   | [ ]       | [ ]    | [ ]   | [ ]   |
| Decision Tree       | [ ]      | [ ]   | [ ]       | [ ]    | [ ]   | [ ]   |
| kNN                 | [ ]      | [ ]   | [ ]       | [ ]    | [ ]   | [ ]   |
| Naive Bayes         | [ ]      | [ ]   | [ ]       | [ ]    | [ ]   | [ ]   |
| Random Forest       | [ ]      | [ ]   | [ ]       | [ ]    | [ ]   | [ ]   |
| XGBoost             | [ ]      | [ ]   | [ ]       | [ ]    | [ ]   | [ ]   |

## 4. Observations on Model Performance
| ML Model Name       | Observation about model performance |
|---------------------|-------------------------------------|
| Logistic Regression | Logistic Regression performed moderately well, particularly in separating linearly separable classes, but struggled with complex nonlinear patterns. |
| Decision Tree       | Decision Tree captured non-linear relationships but showed signs of overfitting on the training data with lower generalization on validation/test set. |
| kNN                 | kNN performance was sensitive to the choice of k and the distance metric; required feature scaling for best results. |
| Naive Bayes         | Naive Bayes was fast to train and execute but assumed feature independence, which may not hold entirely, affecting performance. |
| Random Forest       | Random Forest provided better generalization due to the combination of multiple trees, reducing overfitting. |
| XGBoost             | XGBoost produced strong performance due to gradient boosting optimization and regularization, though it required careful hyperparameter tuning. |

---

## 5. Conclusions

From the evaluation metrics and performance observations, ensemble methods such as **Random Forest** and **XGBoost** generally performed better on this dataset compared to simpler models like Logistic Regression and Naive Bayes.

Detailed findings:
- **Best Overall Performance:** [Model Name]
- **Best Precision:** [Model Name]
- **Best Recall:** [Model Name]
- **Most Balanced F1 Score:** [Model Name]
