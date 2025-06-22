
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE

# Create images directory
os.makedirs("images", exist_ok=True)

# Load dataset
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Data cleaning
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
df = df.drop('customerID', axis=1)

# Feature engineering
df['TenureToChargeRatio'] = df['tenure'] / (df['TotalCharges'] + 1e-5)
df['ServiceDensity'] = (df[['PhoneService', 'StreamingTV', 'StreamingMovies']] == 'Yes').sum(axis=1)

# Churn Pie Chart (Plotly)
fig = px.pie(df, names='Churn', title='Churn Distribution',
             color_discrete_sequence=['#1f77b4', '#ff7f0e'])
fig.write_image("images/churn_pie.png")

# Tenure Violin Plot
plt.figure(figsize=(10,6))
sns.violinplot(x='Churn', y='tenure', data=df, palette='viridis', inner='quartile')
plt.title('Tenure Distribution by Churn Status')
plt.savefig("images/tenure_violin.png", bbox_inches='tight')
plt.close()

# Monthly Charges Box Plot
plt.figure(figsize=(10,6))
sns.boxplot(x='Churn', y='MonthlyCharges', data=df, palette='coolwarm')
plt.title('Monthly Charges by Churn Status')
plt.savefig("images/monthly_charges_box.png", bbox_inches='tight')
plt.close()

# Train/test split and preprocessing
X = df.drop('Churn', axis=1)
y = df['Churn']

cat_cols = X.select_dtypes(include=['object']).columns
num_cols = X.select_dtypes(include=['int64', 'float64']).columns

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
])

X_processed = preprocessor.fit_transform(X)

# Balance data
smote = SMOTE(random_state=42)
X_bal, y_bal = smote.fit_resample(X_processed, y)

# Train XGBoost model
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
X_train, X_test, y_train, y_test = train_test_split(X_bal, y_bal, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix")
plt.savefig("images/confusion_matrix.png", bbox_inches='tight')
plt.close()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.savefig("images/roc_curve.png", bbox_inches='tight')
plt.close()

# Feature Importance
importances = model.feature_importances_
plt.figure(figsize=(10,6))
plt.bar(range(len(importances)), importances)
plt.title("Feature Importances from XGBoost")
plt.xlabel("Feature Index")
plt.ylabel("Importance")
plt.savefig("images/feature_importance.png", bbox_inches='tight')
plt.close()
