import pandas as pd

pd.set_option("display.max_columns", None)   # show all columns, no "..."
pd.set_option("display.width", 200)

columns = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
           "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]

df = pd.read_csv("pima-indians-diabetes.csv")
if df.columns[0] != "Pregnancies":
    df = pd.read_csv("pima-indians-diabetes.csv", header=None, names=columns)

print("=== Shape ===")
print(df.shape)

print("\n=== First 5 rows ===")
print(df.head())

print("\n=== Data types and nulls ===")
print(df.info())

print("\n=== Summary statistics ===")
print(df.describe())

print("\n=== Class balance (1 = diabetic, 0 = not) ===")
print(df["Outcome"].value_counts())
print(df["Outcome"].value_counts(normalize=True).round(3) * 100)

print("\n=== Hidden missing values (zeros that shouldn't be zero) ===")
zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
print((df[zero_cols] == 0).sum())