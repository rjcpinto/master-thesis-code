import duckdb
import os
import sys
from pathlib import Path

DB_DIR = Path("database")

def build_data_warehouse(db_name):
    print(f"Connecting to database: {db_name}...")
    con = duckdb.connect(db_name)

    sql_files = [DB_DIR / "schema/create_dimensions.sql",
                 DB_DIR / "schema/create_fact.sql"]

    for sql_file in sql_files:
        if not os.path.exists(sql_file):
            print(f"Could not find '{sql_file}'. Make sure it is in the same folder as this script.")
            continue
            
        print(f"Reading and executing {sql_file}...")
        with open(sql_file, 'r') as file:
            query = file.read()
            con.execute(query)
            
        print(f"Successfully loaded table from {sql_file}.")

    print("\nData warehouse built successfully! Here are your tables:")
    print(con.execute("SHOW TABLES;").df())
    
    con.close()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 build_database.py <db_name>")
        print("Example: python3 build_database.py data_warehouse.db")
        sys.exit(1)

    db_name = sys.argv[1]
    build_data_warehouse(db_name)