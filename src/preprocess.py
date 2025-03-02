import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import yaml

def preprocess_data():
    # Veriyi yükle
    df = pd.read_csv("data/raw/sales_data.csv")  # PostgreSQL'den çekilen e-commerce dataset
    
    # Eksik verileri temizle
    df = df.dropna()
    
# Aykırı değerleri IQR yöntemi ile filtrele
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

    # Kategorik değişkenleri encode et
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        df[col] = LabelEncoder().fit_transform(df[col])
    
    # Feature scaling
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
    df[numeric_columns] = StandardScaler().fit_transform(df[numeric_columns])
    
    # Train-test ayrimi
    train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)
    
    # Kaydet
    train_data.to_csv("data/processed/train_data.csv", index=False)
    test_data.to_csv("data/processed/test_data.csv", index=False)
    
    print("Preprocessing tamamlandi!")

if __name__ == "__main__":
    preprocess_data()