import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, f1_score
import joblib
import yaml

"""def preprocess_data():
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
    
    print("Preprocessing tamamlandi!")"""

def train_model(model_name, params):
    # Veri yükle
    train_data = pd.read_csv("data/processed/train_data.csv")
    X = train_data.drop(columns=["quantity"])  # Özellikler
    y = train_data["quantity"]  # Hedef değişken
    
    models = {
        "RandomForest": RandomForestRegressor(n_estimators=params["rf_n_estimators"], max_depth=params["rf_max_depth"], random_state=42),
        "XGBoost": XGBRegressor(learning_rate=params["xgboost_learning_rate"], n_estimators=params["xgboost_n_estimators"], max_depth=params["xgboost_max_depth"], random_state=42),
        "LinearRegression": LinearRegression(),
        "LogisticRegression": LogisticRegression(max_iter=params["logistic_max_iter"], solver=params["logistic_solver"], penalty=params["logistic_penalty"])
    }
    if model_name not in models:
        raise ValueError(f"Model {model_name} desteklenmiyor!")
    
    model = models[model_name]
    print(f"Model Eğitiliyor: {model_name}")
    
    model.fit(X, y)
    
    # Tahmin yap
    y_pred = model.predict(X)
    
    # Değerlendirme metrikleri
    if model_name == "LogisticRegression":
        metrics = {"accuracy": accuracy_score(y, y_pred), "f1_score": f1_score(y, y_pred, average="weighted")}
    else:
        metrics = {"mae": mean_absolute_error(y, y_pred), "rmse": np.sqrt(mean_squared_error(y, y_pred))}
    
    
    # Modeli kaydet
    joblib.dump(model, f"models/{model_name}.pkl")
    
    print(f"{model_name} modeli kaydedildi!")
    
    return model, metrics  # Modeli ve metrikleri döndür

"""if __name__ == "__main__":
    preprocess_data()"""

