import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Importing the dataset
df = pd.read_csv("Aman.csv")  # Apne HTS file ka path daal do

# 1. Overview of the dataset
print("Dimensions of the dataset:", df.shape)
print("Columns of the dataset:", df.columns.tolist())
print("Top 5 rows:\n", df.head())
print("Last 5 rows:\n", df.tail())
print("Dataset info:\n")
df.info()
print("\nDescriptive statistics:\n", df.describe())

# 2. Check for missing values
print("Missing values in the dataset:\n", df.isnull().sum())

# 3. Cleaning: Drop completely empty columns
df.drop(columns=["Quota Quantity"], inplace=True)  # Quota Quantity completely empty hai
print("Dataset shape after dropping empty columns:", df.shape)

# 4. Cleaning "General Rate of Duty" to numeric
def clean_duty_rate(rate):
    if pd.isna(rate) or rate == "Free":
        return 0.0
    if isinstance(rate, str):
        rate = rate.replace("¢/kg", "").replace("%", "").strip()
        try:
            return float(rate) / 100 if "%" in rate else float(rate)
        except ValueError:
            return None
    return rate

df["General Rate Numeric"] = df["General Rate of Duty"].apply(clean_duty_rate)

# 5. Basic stats for numeric columns
print("Max values:\n", df[["Indent", "General Rate Numeric"]].max())
print("Min values:\n", df[["Indent", "General Rate Numeric"]].min())
print("Median values:\n", df[["Indent", "General Rate Numeric"]].median())
print("Mean values:\n", df[["Indent", "General Rate Numeric"]].mean())
print("Mode values:\n", df[["Indent", "General Rate Numeric"]].mode().iloc[0])
print("Count of non-null values:\n", df[["Indent", "General Rate Numeric"]].count())

# 6. Histogram of "Indent"
plt.hist(df["Indent"], bins=12, color="blue", edgecolor="black")
plt.xlabel("Indent Level")
plt.ylabel("Frequency")
plt.title("Distribution of Indent Levels")
plt.show()

# 7. Histogram of "General Rate Numeric"
plt.hist(df["General Rate Numeric"].dropna(), bins=20, color="green", edgecolor="black")
plt.xlabel("General Rate of Duty (numeric)")
plt.ylabel("Frequency")
plt.title("Distribution of General Duty Rates")
plt.show()

# 8. Bar chart of top 10 Descriptions by frequency
top_descriptions = df["Description"].value_counts().head(10)
top_descriptions.plot(kind="bar", color="orange")
plt.xlabel("Description")
plt.ylabel("Count")
plt.title("Top 10 Descriptions by Frequency")
plt.xticks(rotation=45, ha="right")
plt.show()

# 9. Correlation Heatmap
numeric_df = df[["Indent", "General Rate Numeric"]].dropna()
corr_matrix = numeric_df.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1, center=0)
plt.title("Correlation Heatmap of Numeric Columns in HTS Data")
plt.show()

# 10. Pivot Table Heatmap (Indent vs General Rate Numeric)
pivot_table = pd.pivot_table(df, values="General Rate Numeric", index="Indent", aggfunc="mean").fillna(0)
plt.figure(figsize=(10, 6))
sns.heatmap(pivot_table, annot=True, cmap="YlGnBu", fmt=".2f")
plt.title("Heatmap of Average General Rate of Duty by Indent Level")
plt.xlabel("Average Duty Rate")
plt.ylabel("Indent Level")
plt.show()
