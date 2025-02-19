import pandas as pd
import psycopg2
from sqlalchemy import create_engine
# PostgreSQL bağlanti bilgilerini gir
DB_NAME = "e_commerce_db"
DB_USER = "Merve Sena"  # Kendi kullanici adini yaz
DB_PASS = "1453"  # Şifreni buraya ekle
DB_HOST = "localhost"
DB_PORT = "5432"

# SQLAlchemy kullanarak bağlanti oluştur
engine = create_engine(f'postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}')
conn = engine.connect()
print("Successfuly Connection PostgreSQL DB")
