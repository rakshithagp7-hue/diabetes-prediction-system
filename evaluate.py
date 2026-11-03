import os
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve,
                             confusion_matrix, classification_report)

columns = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
           "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]

df = pd.read_csv("pima-indians-diabetes.csv")
if df.columns[0] != "Pregnancies":
    df = pd.read_csv("pima-indians-diabetes.csv", header=None, names=columns)

# Rebuild the exact same test set (same settings as train_model.py)
zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df[zero_cols] = df[zero_cols].replace(0, np.nan)
X = df.drop("Outcome", axis=1)
y = df["Outcome"]
_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Load saved model, fill and scale the test data the same way as training
saved = joblib.load("model.pkl")
model, scaler, medians = saved["model"], saved["scaler"], saved["medians"]
X_test = X_test.fillna(medians)
X_test_scaled = scaler.transform(X_test)

y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

# --- Metrics ---
print("=== Test Set Metrics ===")
print(f"Accuracy : {accuracy_score(y_test, y_pred):.3f}")
print(f"Precision: {precision_score(y_test, y_pred):.3f}")
print(f"Recall   : {recall_score(y_test, y_pred):.3f}")
print(f"F1-Score : {f1_score(y_test, y_pred):.3f}")
print(f"ROC-AUC  : {roc_auc_score(y_test, y_prob):.3f}")

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred,
                            target_names=["Non-Diabetic", "Diabetic"]))

cm = confusion_matrix(y_test, y_pred)
print("=== Confusion Matrix ===")
print(cm)

os.makedirs("plots", exist_ok=True)

# Confusion matrix plot
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Non-Diabetic", "Diabetic"],
            yticklabels=["Non-Diabetic", "Diabetic"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("plots/5_confusion_matrix.png", dpi=150)
plt.close()

# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"Random Forest (AUC = {roc_auc_score(y_test, y_prob):.3f})")
plt.plot([0, 1], [0, 1], "k--", label="Random guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig("plots/6_roc_curve.png", dpi=150)
plt.close()

# Feature importance plot
imp = pd.Series(model.feature_importances_, index=saved["features"]).sort_values()
plt.figure(figsize=(7, 4))
imp.plot(kind="barh", color="teal")
plt.title("Feature Importance")
plt.tight_layout()
plt.savefig("plots/7_feature_importance.png", dpi=150)
plt.close()

print("\nSaved 3 charts in the 'plots' folder.")