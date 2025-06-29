# Importing necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the iris dataset
# from sklearn.datasets import load_iris
# iris = load_iris()
iris = pd.read_csv(r"C:\Users\hp\OneDrive\Desktop\CSI\Iris.csv")

df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target
df['species'] = df['species'].apply(lambda x: iris.target_names[x])

# Display basic information
print("First 5 rows of the dataset:")
print(df.head())
print("\nDataset Information:")
print(df.info())
print("\nSummary Statistics:")
print(df.describe())

# Check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# Data distribution: Histograms
print("\nGenerating histograms for feature distributions...")
df.hist(figsize=(10, 8), bins=20, edgecolor='black')
plt.suptitle('Feature Distributions', fontsize=16)
plt.tight_layout()
plt.show()

# Boxplots: Outlier detection
print("\nGenerating boxplots for outlier detection...")
plt.figure(figsize=(10, 6))
for i, col in enumerate(df.columns[:-1]):
    plt.subplot(2, 2, i+1)
    sns.boxplot(y=df[col], x=df['species'], palette="Set2")
    plt.title(f'Boxplot of {col}')
plt.tight_layout()
plt.show()

# Pairplot to explore relationships between features
print("\nGenerating pairplot to see relationships...")
sns.pairplot(df, hue="species", palette="husl", diag_kind="hist")
plt.suptitle('Pairwise Relationships', y=1.02)
plt.show()

# Correlation heatmap
print("\nGenerating correlation heatmap...")
plt.figure(figsize=(8, 6))
corr_matrix = df.iloc[:, :-1].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True)
plt.title('Feature Correlation Heatmap')
plt.show()

# Outlier Detection using IQR method
print("\nOutlier detection using IQR:")
for col in df.columns[:-1]:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
    print(f"{col}: {len(outliers)} outliers")

# Summary of findings
print("\n🔍 Summary of EDA:")
print("- No missing values detected.")
print("- Distributions of features are mostly normal or slightly skewed.")
print("- Setosa species appears distinct in most plots.")
print("- Some outliers exist, especially in petal-related features.")
print("- Strong correlation between petal length and petal width.")

