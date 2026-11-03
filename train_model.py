import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

columns = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
           "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]

df = pd.read_csv("pima-indians-diabetes.csv")
if df.columns[0] != "Pregnancies":
    df = pd.read_csv("pima-indians-diabetes.csv", header=None, names=columns)

# --- Preprocessing (same as Step 3) ---
zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df[zero_cols] = df[zero_cols].replace(0, np.nan)

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

medians = X_train.median()
X_train = X_train.fillna(medians)
X_test = X_test.fillna(medians)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Train Random Forest ---
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    min_samples_leaf=3,
    class_weight="balanced",   # helps with the 65/35 imbalance
    random_state=42,
)
model.fit(X_train_scaled, y_train)

# --- Quick check ---
train_acc = accuracy_score(y_train, model.predict(X_train_scaled))
test_acc = accuracy_score(y_test, model.predict(X_test_scaled))
cv = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring="accuracy")

print(f"Train accuracy:        {train_acc:.3f}")
print(f"Test accuracy:         {test_acc:.3f}")
print(f"5-fold CV accuracy:    {cv.mean():.3f} (+/- {cv.std():.3f})")

print("\nFeature importance:")
imp = pd.Series(model.feature_importances_, index=X.columns)
print(imp.sort_values(ascending=False).round(3))

# --- Save everything the app needs ---
joblib.dump(
    {"model": model, "scaler": scaler, "medians": medians,
     "features": list(X.columns)},
    "model.pkl",
)
print("\nSaved model.pkl")