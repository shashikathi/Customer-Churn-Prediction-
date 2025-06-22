
# 🧠 Customer Churn Prediction

> _“Churn is expensive. Predict it before it happens.”_

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-orange?logo=Jupyter)
![XGBoost](https://img.shields.io/badge/XGBoost-🔥%20Classifier-orange?logo=xgboost)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-yellow?logo=scikit-learn)
![Seaborn](https://img.shields.io/badge/Seaborn-Charts-teal?logo=seaborn)
![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Viz-lightblue?logo=plotly)
![Status](https://img.shields.io/badge/Project-Production%20Ready-brightgreen?style=flat-square)

---


---

## 🔍 TL;DR

Machine Learning pipeline that predicts whether telecom customers are likely to churn.  
It’s fast, visual, and designed for real-world business action.

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
```

---

## 🤖 Use the Trained Model Instantly

```python
import joblib
model = joblib.load("churn_prediction_pipeline.pkl")
preds = model.predict(X_new)
```

Or use the ready-made file: `churn_predictions_with_probs.csv`

---

## 💡 Real-World Impact

> 📉 Reduce churn by 5-15%  
> 📈 Boost revenue retention  
> 🧠 Explainable ML = Better business decisions  

---

## 🔧 Tech Stack

| Purpose              | Libraries Used                                     |
|----------------------|----------------------------------------------------|
| 📊 Data & Viz         | `pandas`, `seaborn`, `matplotlib`, `plotly`        |
| 🤖 ML Pipeline        | `scikit-learn`, `xgboost`, `joblib`, `SMOTE`       |
| 📁 Export + Serving   | `.pkl`, `.csv`, `images/` folder for portability    |

---

## 👀 Live Look at What’s Inside

> ![ml](https://media.giphy.com/media/WFZvB7VIXBgiz3oDXE/giphy.gif)

---

## 🙏 Credits

- [Kaggle Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)  
- [XGBoost](https://xgboost.readthedocs.io/)  
- [SMOTE (imbalanced-learn)](https://imbalanced-learn.org/stable/)

---

## 🤝 Let’s Connect

- 🌐 [My Portfolio](https://kshashi-preetham-5tbnyvy.gamma.site/)  
- 💼 [LinkedIn](https://linkedin.com/in/shashikathi)  
- 🧠 Open to collaboration, feedback, and freelance gigs!

---

> ⭐ If this saved you time, consider giving it a star.  
> 💬 Got feedback? Drop an issue or connect with me directly.
