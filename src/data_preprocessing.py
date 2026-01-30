import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

columns = [
    'age','sex','cp','trestbps','chol','fbs','restecg',
    'thalach','exang','oldpeak','slope','ca','thal','target'
]

df = pd.read_csv(
    'data/heart.csv',
    names=columns
)


print(df.shape)
print(df.head())
# Replace '?' with NaN
df.replace('?', np.nan, inplace=True)

# Count missing values
print("Missing values per column:")
print(df.isnull().sum())

# Convert columns to numeric (required because '?' are strings)
for col in ['ca','thal']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Fill missing values with the mode (most common value)
for col in ['ca','thal']:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Verify no more missing values
print("\nMissing values after cleaning:")
print(df.isnull().sum())

# Convert multi-class target to binary
df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)


# exploratory data analysis
import matplotlib.pyplot as plt
import seaborn as sns

# Distribution of target
sns.countplot(x='target', data=df)
plt.title('Heart Disease Distribution')
plt.show()

# Correlation heatmap
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation')
plt.show()

# Age distribution
sns.histplot(df['age'], kde=True, bins=20)
plt.title('Age Distribution')
plt.show()

# Chest pain vs target
sns.countplot(x='cp', hue='target', data=df)
plt.title('Chest Pain Type vs Heart Disease')
plt.show()





# def preprocess_data(path):
#     df = pd.read_csv(path)

#     X = df.drop('target', axis=1)
#     y = df['target']

#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)

#     X_train, X_test, y_train, y_test = train_test_split(
#         X_scaled, y, test_size=0.2, random_state=42
#     )

#     return X_train, X_test, y_train, y_test, scaler

df.to_csv('data/heart_cleaned.csv', index=False)
print("Cleaned dataset saved as heart_cleaned.csv")


def preprocess_data(path):
    df = pd.read_csv(path)

    X = df.drop('target', axis=1)
    y = df['target']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test, scaler