import mlflow
import yaml
import train

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Ecommerce Sales Analysis")

with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)["train"]

models = ["RandomForest", "XGBoost", "LinearRegression"]

for model_name in models:
    print(f"{model_name} modeli eğitiliyor...")
    trained_model, metrics = train.train_model(model_name)
    
    if trained_model is None:
        print(f" {model_name} eğitilemedi!")
    else:
        with mlflow.start_run(run_name=model_name):
            mlflow.log_metrics(metrics)
            mlflow.log_param("model_name", model_name)
            mlflow.sklearn.log_model(trained_model, model_name)
            mlflow.end_run()
            print(f" {model_name} başariyla MLflow'a kaydedildi!")
