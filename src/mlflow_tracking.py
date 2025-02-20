import mlflow
import mlflow.sklearn
import yaml
import pickle
from sklearn.metrics import accuracy_score, f1_score
from train import train_model  # Eğitim fonksiyonu çağrılıyor

# MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:5000")

# Deney adı
mlflow.set_experiment("Ecommerce Sales Analysis")

# Parametreleri oku
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)["train"]

# Kullanılacak modeller
models = ["RandomForest", "XGBoost", "LogisticRegression"]

# Modelleri eğit ve MLflow'a kaydet
for model_name in models:
    with mlflow.start_run(run_name=model_name):
        print(f"Model Eğitiliyor: {model_name}")
        
        model, X_test, y_test = train_model(model_name)
        y_pred = model.predict(X_test)
        
        # Değerlendirme metrikleri
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred, average="weighted")
        }
        
        # MLflow logları
        mlflow.log_metrics(metrics)
        mlflow.log_param("model_name", model_name)
        
        # Eğitilmiş modeli kaydet
        model_path = f"models/{model_name}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        
        mlflow.sklearn.log_model(model, model_name)
        
        print(f"{model_name} modeli MLflow'a kaydedildi! ")
