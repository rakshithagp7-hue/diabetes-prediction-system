import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # save images without opening windows
import matplotlib.pyplot as plt
import seaborn as sns

columns = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
           "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]

df = pd.read_csv("pima-indians-diabetes.csv")
if df.columns[0] != "Pregnancies":
    df = pd.read_csv("pima-indians-diabetes.csv", header=None, names=columns)

# Clean for plotting: zeros in these columns are missing, fill with median
zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df[zero_cols] = df[zero_cols].replace(0, np.nan)
df[zero_cols] = df[zero_cols].fillna(df[zero_cols].median())

os.makedirs("plots", exist_ok=True)
sns.set_style("whitegrid")
labels = {0: "Non-Diabetic", 1: "Diabetic"}
df["Status"] = df["Outcome"].map(labels)

# 1. Class distribution
plt.figure(figsize=(5, 4))
sns.countplot(data=df, x="Status", hue="Status", palette="Set2", legend=False)
plt.title("Class Distribution")
plt.tight_layout()
plt.savefig("plots/1_class_distribution.png", dpi=150)
plt.close()

# 2. Correlation heatmap
plt.figure(figsize=(9, 7))
corr = df.drop(columns="Status").corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("plots/2_correlation_heatmap.png", dpi=150)
plt.close()

# 3. Glucose, BMI, Age distributions by outcome
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col in zip(axes, ["Glucose", "BMI", "Age"]):
    sns.histplot(data=df, x=col, hue="Status", kde=True, ax=ax,
                 palette="Set2", element="step")
    ax.set_title(f"{col} by Diabetes Status")
plt.tight_layout()
plt.savefig("plots/3_distributions.png", dpi=150)
plt.close()

# 4. Boxplots
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col in zip(axes, ["Glucose", "BMI", "Age"]):
    sns.boxplot(data=df, x="Status", y=col, hue="Status",
                palette="Set2", legend=False, ax=ax)
    ax.set_title(f"{col} vs Outcome")
plt.tight_layout()
plt.savefig("plots/4_boxplots.png", dpi=150)
plt.close()

# Print correlation with Outcome
print("Correlation of each feature with Outcome:")
print(corr["Outcome"].drop("Outcome").sort_values(ascending=False).round(3))
print("\nSaved 4 charts in the 'plots' folder.")