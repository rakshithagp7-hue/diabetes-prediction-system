import pandas as pd

columns = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
           "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]

df = pd.read_csv("pima-indians-diabetes.csv")

# If the file has no header row, the first data row becomes the column names.
# Detect that and reload with proper names.
if df.columns[0] != "Pregnancies":
    df = pd.read_csv("pima-indians-diabetes.csv", header=None, names=columns)

print(df.shape)
print(df.head())