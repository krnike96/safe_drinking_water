import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# To handle class imbalance (SMOTE)
from imblearn.over_sampling import SMOTE

# 1. DATA LOADING AND INITIAL INSPECTION
print("--- 1. Loading Data ---")
try:
    df = pd.read_csv('Dataset/water_potability.csv')
    print("Initial dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'water_potability.csv' not found. Please place it in the project directory.")
    exit()

# Separate features (X) and target (y)
X = df.drop('Potability', axis=1)
y = df['Potability']

# 2. DATA CLEANING AND IMPUTATION
# Use SimpleImputer to fill missing values (NaN) with the mean of the column
print("\n--- 2. Data Cleaning and Imputation ---")
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
# Fit and transform the imputer on the feature set X
X_imputed_array = imputer.fit_transform(X)
X_imputed = pd.DataFrame(X_imputed_array, columns=X.columns)

print("Missing values imputed using the mean for all features.")

# 3. FEATURE SCALING
# Standardize features (Scaling is vital for ML models)
print("\n--- 3. Feature Scaling ---")
scaler = StandardScaler()
X_scaled_array = scaler.fit_transform(X_imputed)
X_scaled = pd.DataFrame(X_scaled_array, columns=X.columns)

# Save the scaler for deployment
joblib.dump(scaler, 'scaler.pkl')
print("Features scaled and StandardScaler saved as 'scaler.pkl'.")

# 4. DATA SPLITTING (TRAINING AND TESTING SETS)
print("\n--- 4. Data Splitting ---")
# Split data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training set size: {len(X_train)} | Testing set size: {len(X_test)}")