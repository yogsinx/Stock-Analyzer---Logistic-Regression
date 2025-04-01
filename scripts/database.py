from config.db_config import get_db_connection

def create_stock_table():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS stock_data (
            id SERIAL PRIMARY KEY,
            date DATE,
            open FLOAT,
            high FLOAT,
            low FLOAT,
            close FLOAT,
            volume BIGINT
        );
    ''')
    conn.commit()
    cursor.close()
    conn.close()
    print("Stock table created!")

if __name__ == "__main__":
    create_stock_table()