# HeartDiseasePrediction

## 1. Problem Statement

The objective of this project is to build and evaluate multiple machine learning classification models to predict whether a customer will opt for a subscription service based on their shopping behavior and demographic attributes.

This project demonstrates an end-to-end machine learning workflow, including:

Data preprocessing

Feature engineering

Model training and evaluation

Comparison of multiple classification algorithms

Deployment of a trained model using a Streamlit web application

The final goal is to assist businesses in identifying customers who are more likely to subscribe, enabling better marketing and retention strategies.

## 2. Dataset Description

Dataset Name: Customer Shopping Behavior Dataset

Source: Public dataset (Kaggle / UCI-style dataset)

Number of Instances: 3900

Number of Features: 17 (excluding target variable)

Target Variable: Subscription Status (Yes / No)

## 3. Machine Learning Models Used

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

| ML Model Name       | Accuracy |    AUC   | Precision |  Recall  |    F1    |   MCC    |
|---------------------|----------|----------|-----------|----------|----------|----------|
| Logistic Regression | 0.847877 | 0.699098 | 0.500000  | 0.069767 | 0.122449 | 0.142646 |
| Decision Tree       | 0.739387 | 0.521897 | 0.184932  | 0.209302 | 0.196364 | 0.041662 |
| kNN                 | 0.829009 | 0.589347 | 0.264706  | 0.069767 | 0.110429 | 0.064067 |
| Naive Bayes         | 0.806604 | 0.680909 | 0.253521  | 0.139535 | 0.180000 | 0.085347 |
| Random Forest       | 0.843160 | 0.639513 | 0.357143  | 0.038760 | 0.069930 | 0.073963 |
| XGBoost             | 0.838443 | 0.631551 | 0.366667  | 0.085271 | 0.138365 | 0.114403 |

## 4. Observations on Model Performance
| ML Model Name       | Observation about model performance |
|---------------------|-------------------------------------|
| Logistic Regression | Logistic Regression performs well due to linear separability. |
| Decision Tree       | Decision Tree captures non-linear patterns but may overfit.   |
| kNN                 | kNN performance depends on distance metric and k value.       |
| Naive Bayes         | Naive Bayes works fast but assumes feature independence. |
| Random Forest       | Random Forest gives better generalization due to ensemble learning and reduce overfitting. |
| XGBoost             | XGBoost provide a strong performance by boosting weak learners and optimizing errors iteratively. |

---

## 5. Conclusions

From the evaluation metrics and performance observations, ensemble methods such as **Random Forest** and **XGBoost** generally performed better on this dataset compared to simpler models like Logistic Regression and Naive Bayes.

Detailed findings:
Best Overall Performance: Logistic Regression

Has the highest accuracy among all models (0.8479) and also maintains the highest AUC (0.6991), indicating better overall discrimination.

Best Precision: Logistic Regression

Highest precision (0.5000), meaning among predicted positives it has the best correctness.

Best Recall: Decision Tree

Highest recall (0.2093), meaning it correctly captures more of the actual positive class than others.

Most Balanced F1 Score: Naive Bayes

Naive Bayes has the best F1 score compared to other models except Logistic Regression

Although Logistic Regression's F1 isn't far behind, Naive Bayes has a more balanced trade-off between precision and recall.
