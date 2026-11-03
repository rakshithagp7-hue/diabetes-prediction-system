import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

columns = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
           "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]

df = pd.read_csv("pima-indians-diabetes.csv")
if df.columns[0] != "Pregnancies":
    df = pd.read_csv("pima-indians-diabetes.csv", header=None, names=columns)

# 1. Handle missing values: zeros in these columns mean "missing"
zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df[zero_cols] = df[zero_cols].replace(0, np.nan)
print("Missing values before filling:")
print(df.isnull().sum())

# 2. Split first, then compute medians from the training set only
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

medians = X_train.median()
X_train = X_train.fillna(medians)
X_test = X_test.fillna(medians)

print("\nMissing values after filling:")
print(X_train.isnull().sum().sum(), "in train,", X_test.isnull().sum().sum(), "in test")

# 3. Scale features (fit on train only)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nTrain size:", X_train.shape, " Test size:", X_test.shape)
print("Diabetic share in train:", round(y_train.mean(), 3))
print("Diabetic share in test: ", round(y_test.mean(), 3))
print("\nScaled train means (should be ~0):")
print(np.round(X_train_scaled.mean(axis=0), 3))