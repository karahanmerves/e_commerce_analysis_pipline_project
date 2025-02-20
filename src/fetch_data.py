import pandas as pd
from db_connection import get_db_engine  # Bağlantıyı içe aktarıyoruz

def fetch_and_save_data():
    #PostgreSQL'den veriyi çek ve CSV olarak kaydet.#
    engine = get_db_engine()  # Bağlantıyı al
    query = "SELECT * FROM sales_data"  

    # SQL sorgusunu çalıştır ve veriyi al
    df = pd.read_sql(query, engine)

    # Veriyi kaydet
    df.to_csv("data/raw/sales_data.csv", index=False)
    print("Data fetched and saved to data/raw/sales_data.csv")

if __name__ == "__main__":
    fetch_and_save_data()
