import pandas as pd
from sklearn.model_selection import train_test_split

# Ham veriyi yükle
data = pd.read_csv("data/raw/Iris.csv")

#Ekstra özellik ekleme
data["SepalRatio"] = data["SepalLengthCm"] / data["SepalWidthCm"]

# İlk 10 satırı silelim
data = data.iloc[10:]

# Train-test ayrımı
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# İşlenmiş verileri kaydet
train_data.to_csv("data/processed/train_data.csv", index=False)
test_data.to_csv("data/processed/test_data.csv", index=False)

print("Preprocessing tamamlandi. Veriler kaydedildi: data/processed/train_data.csv, data/processed/test_data.csv")


