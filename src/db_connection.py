import pandas as pd
import psycopg2
from sqlalchemy import create_engine

# PostgreSQL bağlanti bilgileri
DB_NAME = "e_commerce_db"
DB_USER = "Merve Sena"  
DB_HOST = "localhost"
DB_PORT = "5432"

def get_db_engine():
    """PostgreSQL bağlantisini döndürür."""
    engine = create_engine(f'postgresql://{DB_USER}:{1453}@{DB_HOST}:{DB_PORT}/{DB_NAME}')
    return engine

if __name__ == "__main__":
    engine = get_db_engine()
    print("Successfully connected to PostgreSQL DB")

