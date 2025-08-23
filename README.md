# 🧠 Customer Churn Prediction

> _“Churn is expensive. Predict it before it happens.”_

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-orange?logo=Jupyter)
![XGBoost](https://img.shields.io/badge/XGBoost-🔥%20Classifier-orange?logo=xgboost)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-yellow?logo=scikit-learn)
![Seaborn](https://img.shields.io/badge/Seaborn-Charts-teal?logo=seaborn)
![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Viz-lightblue?logo=plotly)
![Status](https://img.shields.io/badge/Project-Production%20Ready-brightgreen?style=flat-square)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-HuggingFace-blue?logo=huggingface)](https://huggingface.co/spaces/shashikathi56/customer-churn-predictor)

---

## 🔍 TL;DR

Machine Learning pipeline that predicts whether telecom customers are likely to churn.  
It’s fast, visual, and designed for real-world business action.

---

## 🚀 Try it Live

👉 [Customer Churn Predictor (Hugging Face Space)](https://huggingface.co/spaces/shashikathi56/customer-churn-predictor)  

No setup needed. Play with the model directly in your browser.

---

## 🧰 Features

- 🧼 Data cleaning & feature engineering  
- ⚖️ Class balancing with SMOTE  
- 🔍 XGBoost with high accuracy (>85%)  
- 📈 Visuals saved to `/images/` — perfect for decks  
- 📦 Trained model + pre-generated predictions

---

## 🗂 Folder Structure

| File / Folder                        | Description                             |
|-------------------------------------|-----------------------------------------|
| `Customer_Churn_Prediction_.ipynb`  | Full ML pipeline in Jupyter             |
| `churn_prediction_pipeline.pkl`     | ✅ Trained XGBoost model                 |
| `churn_predictions_with_probs.csv`  | 📄 Predictions with churn probabilities |
| `images/`                           | 📸 Saved visuals (charts, ROC, etc.)     |
| `requirements.txt`                  | All needed libraries                    |

---

## 🖼 Visual Output

| Chart Type            | 📁 File                     |
|------------------------|----------------------------|
| 🥧 Churn Distribution   | `churn_pie.png`            |
| 🎻 Tenure Violin Plot   | `tenure_violin.png`        |
| 📦 Charges Boxplot      | `monthly_charges_box.png`  |
| 📊 Confusion Matrix     | `confusion_matrix.png`     |
| 📉 ROC Curve            | `roc_curve.png`            |
| 🌟 Feature Importance   | `feature_importance.png`   |

---

## 🚀 Quickstart

```bash
git clone https://github.com/yourusername/customer-churn-prediction.git
cd customer-churn-prediction
pip install -r requirements.txt
jupyter notebook Customer_Churn_Prediction_.ipynb
