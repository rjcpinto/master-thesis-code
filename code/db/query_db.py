import duckdb
import pandas as pd
import sys
from prompt_toolkit import prompt
from pathlib import Path

OUT_DIR = Path("datasets_final")
DB_DIR = Path("database")

def query_database(con, ml_query, output_name, is_show_tables):
    try:
        ml_dataframe = con.execute(ml_query).df()

        print(f"\n Query successful! Extracted {len(ml_dataframe)} rows and {len(ml_dataframe.columns)} features.")
        if is_show_tables == "yes":
            print(ml_dataframe)
        else:
            print(ml_dataframe.head())
                
        want_save = input("\nDo you want to save this dataset to a CSV file? (yes/no): ").strip().lower()
                
        if want_save == 'yes':
            out_path = OUT_DIR / f"{output_name}.csv"
                    
            ml_dataframe.to_csv(
                out_path, 
                index=False,
                sep='\t',
                na_rep='NA',
                encoding='utf-8'
            )
            
            print(f"Dataset saved to: {out_path}")
            return True  
        else:
            print("Dataset discarded. Let's try writing a new query.")
            return False 
                
    except Exception as e:
        print(f"\n Error executing SQL query:\n{e}")
        print("Please fix the query and try again.")
        return False

def extract_ml_dataset(output_name, db_name):
    db_path = DB_DIR / f"{db_name}.db"
    
    try:
        con = duckdb.connect(db_path, read_only=True)
        print(f" Connected to the data warehouse: {db_name}.db")
    except Exception as e:
        print(f"Failed to connect to database: {e}")
        return

    while True:
        print("\n" + "="*50)
        
        print("Input your SQL query. You can use UP/DOWN arrows to navigate lines freely.")
        print("When finished, press [Esc] followed by [Enter] to execute.")
        print("Type 'exit' to quit.")
        print("Helpful commands: 'SHOW TABLES;', 'DESCRIBE table_name;', 'SELECT * FROM information_schema.columns;'")
        
        ml_query = prompt("\n> ", multiline=True)
        
        if ml_query.replace(';', '').strip().lower() == 'exit':
            print("Exiting...")
            con.close()
            return
            
        if not ml_query.replace(';', '').strip():
            continue

        ml_query_lower = ml_query.strip().lower()
        if ml_query_lower.startswith('show') or ml_query_lower.startswith('describe') or 'information_schema' in ml_query_lower:
            should_break = query_database(con, ml_query, output_name, is_show_tables="yes")
        else:
            should_break = query_database(con, ml_query, output_name, is_show_tables="no")
            
        if should_break:
            break

    con.close()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 query_db.py [output_name] [db_name]")
        print("Example: python3 query_db.py ml_dataset data_warehouse")
        sys.exit(1)

    output_name = sys.argv[1]
    db_name = sys.argv[2]  
    extract_ml_dataset(output_name, db_name)