import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import mlflow
import mlflow.sklearn

# MLflow oturumunu başlat
mlflow.start_run()

# İşlenmiş veriyi yükle
train_data = pd.read_csv("data/processed/train_data.csv")

# Özellikler (features) ve etiket (target) ayırma
X = train_data.drop(columns=["Id", "Species"])  # Özellikler
y = train_data["Species"]  # Etiket

# Modeli oluştur
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Modeli eğit
model.fit(X, y)

# Modeli kaydet
joblib.dump(model, "models/model.pkl")

# Eğitim sonrası doğruluk ölçümü
y_pred = model.predict(X)
accuracy = accuracy_score(y, y_pred)
print(f"Model doğruluğu: {accuracy:.4f}")

# MLflow ile metrikleri kaydet
mlflow.log_metric("accuracy", accuracy)

# Modeli MLflow ile kaydet
mlflow.sklearn.log_model(model, "model")

# MLflow oturumunu bitir
mlflow.end_run()
