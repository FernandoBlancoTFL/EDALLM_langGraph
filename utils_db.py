import sqlite3

def save_to_db(df, db_name="ncr_bookings.db", table_name="bookings"):
    """
    Save a pandas DataFrame to a SQLite database.
    """
    conn = sqlite3.connect(db_name)
    df.to_sql(table_name, conn, if_exists="replace", index=False)
    conn.close()
    print(f"✅ DataFrame saved to {db_name}, table: {table_name}")
