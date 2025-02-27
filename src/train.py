#import pdb; pdb.set_trace()#
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import yaml
import os

# Parametreleri `params.yaml` dosyasından yükle
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)["train"]

def train_model(model_name):
    # Veri yükle
    train_data = pd.read_csv("data/processed/train_data.csv")
    X = train_data.drop(columns=["quantity"])  # Özellikler
    y = train_data["quantity"]  # Hedef değişken
    
    models = {
        "RandomForest": RandomForestRegressor(n_estimators=params["rf_n_estimators"], max_depth=params["rf_max_depth"], random_state=42),
        "XGBoost": XGBRegressor(learning_rate=params["xgboost_learning_rate"], n_estimators=params["xgboost_n_estimators"], max_depth=params["xgboost_max_depth"], random_state=42),
        "LinearRegression": LinearRegression(),
    }
    
    if model_name not in models:
        raise ValueError(f"Model {model_name} desteklenmiyor!")
    
    model = models[model_name]
    print(f"Model Eğitiliyor: {model_name}")
    
    model.fit(X, y)
    
    # Tahmin yap
    y_pred = model.predict(X)
    
    # Değerlendirme metrikleri
    metrics = {"mae": mean_absolute_error(y, y_pred), "rmse": np.sqrt(mean_squared_error(y, y_pred))}
        
    if not os.path.exists("models"):
        os.makedirs("models")
    # Modeli kaydet
    joblib.dump(model, f"models/{model_name}.pkl")
    print(f"{model_name} modeli başariyla kaydedildi: models/{model_name}.pkl", flush=True)

    
    return model, metrics  # Modeli ve metrikleri döndür

if __name__ == "__main__":
    print("train.py başladi!", flush=True)

    # Model isimlerini tanımla ve eğit
    models = ["RandomForest", "XGBoost", "LinearRegression"]
    for model_name in models:
        print(f"🛠 {model_name} modeli eğitiliyor...", flush=True)
        trained_model, metrics = train_model(model_name)
        print(f"{model_name} eğitildi: {metrics}", flush=True)
