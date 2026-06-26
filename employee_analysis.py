import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
# -----------------------------
# BASIC INFORMATION
# -----------------------------
print("=" * 50)
print("EMPLOYEE ATTRITION ANALYSIS")
print("=" * 50)

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 Rows:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nData Types:")
print(df.dtypes)

# -----------------------------
# DATA CLEANING
# -----------------------------
df = df.drop_duplicates()

print("\nShape After Removing Duplicates:")
print(df.shape)

# -----------------------------
# ATTRITION COUNT
# -----------------------------
plt.figure(figsize=(6,4))
sns.countplot(x='Attrition', data=df)
plt.title("Employee Attrition Count")
plt.show()

# -----------------------------
# DEPARTMENT VS ATTRITION
# -----------------------------
plt.figure(figsize=(8,5))
sns.countplot(x='Department', hue='Attrition', data=df)
plt.xticks(rotation=20)
plt.title("Department Wise Attrition")
plt.show()

# -----------------------------
# OVERTIME VS ATTRITION
# -----------------------------
plt.figure(figsize=(6,4))
sns.countplot(x='OverTime', hue='Attrition', data=df)
plt.title("Overtime vs Attrition")
plt.show()

# -----------------------------
# JOB SATISFACTION VS ATTRITION
# -----------------------------
plt.figure(figsize=(6,4))
sns.countplot(x='JobSatisfaction', hue='Attrition', data=df)
plt.title("Job Satisfaction vs Attrition")
plt.show()

# -----------------------------
# SALARY VS ATTRITION
# -----------------------------
plt.figure(figsize=(8,5))
sns.boxplot(x='Attrition', y='MonthlyIncome', data=df)
plt.title("Monthly Income vs Attrition")
plt.show()

# -----------------------------
# AGE DISTRIBUTION
# -----------------------------
plt.figure(figsize=(8,5))
sns.histplot(df['Age'], bins=20, kde=True)
plt.title("Age Distribution")
plt.show()

# -----------------------------
# YEARS AT COMPANY
# -----------------------------
plt.figure(figsize=(8,5))
sns.histplot(df['YearsAtCompany'], bins=15, kde=True)
plt.title("Years at Company Distribution")
plt.show()

# -----------------------------
# CORRELATION HEATMAP
# -----------------------------
numeric_df = df.select_dtypes(include=['int64', 'float64'])

plt.figure(figsize=(12,8))
sns.heatmap(numeric_df.corr(), cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# -----------------------------
# ATTRITION RATE
# -----------------------------
attrition_rate = (
    df[df['Attrition'] == 'Yes'].shape[0]
    / df.shape[0]
) * 100

print("\nAttrition Rate: {:.2f}%".format(attrition_rate))

# -----------------------------
# KEY INSIGHTS
# -----------------------------
print("\nPROJECT COMPLETED SUCCESSFULLY")
# Attrition Count
print("\nAttrition Count:")
print(df['Attrition'].value_counts())

# Department-wise Attrition
print("\nDepartment-wise Attrition:")
print(pd.crosstab(df['Department'], df['Attrition']))

# Overtime vs Attrition
print("\nOvertime vs Attrition:")
print(pd.crosstab(df['OverTime'], df['Attrition']))

# Average Salary by Attrition
print("\nAverage Monthly Income:")
print(df.groupby('Attrition')['MonthlyIncome'].mean())

# Job Satisfaction by Attrition
print("\nJob Satisfaction:")
print(pd.crosstab(df['JobSatisfaction'], df['Attrition']))
plt.show()
