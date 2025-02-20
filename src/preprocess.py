import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import joblib
import yaml

def preprocess_data():
    # Veriyi yükle
    df = pd.read_csv("data/raw/sales_data.csv")  # PostgreSQL'den çekilen e-commerce dataset
    
    # Eksik verileri temizle
    df = df.dropna()
    
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

def train_model():
    # MLflow başlat
    mlflow.set_experiment("Ecommerce Sales Analysis")
    
    # Parametreleri oku
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)["train"]
    
    # Veri yükle
    train_data = pd.read_csv("data/processed/train_data.csv")
    X = train_data.drop(columns=["target"])  # Özellikler
    y = train_data["target"]  # Hedef değişken
    
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=params["rf_n_estimators"], max_depth=params["rf_max_depth"], random_state=42),
        "XGBoost": XGBClassifier(learning_rate=params["xgboost_learning_rate"], n_estimators=params["xgboost_n_estimators"], max_depth=params["xgboost_max_depth"], random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=500)
    }
    
    for model_name, model in models.items():
        with mlflow.start_run(run_name=model_name):
            print(f"Model Eğitiliyor: {model_name}")
            
            model.fit(X, y)
            
            # Tahmin yap
            y_pred = model.predict(X)
            
            # Değerlendirme metrikleri
            metrics = {
                "accuracy": accuracy_score(y, y_pred),
                "f1_score": f1_score(y, y_pred, average="weighted")
            }
            
            # MLflow loglari
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, model_name)
            joblib.dump(model, f"models/{model_name}.pkl")
            
            print(f"✅ {model_name} modeli MLflow'a kaydedildi!")

if __name__ == "__main__":
    preprocess_data()
    train_model()
