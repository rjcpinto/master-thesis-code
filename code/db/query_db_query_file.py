import duckdb
import pandas as pd
import sys
from prompt_toolkit import prompt

def query_database(query_file, output_name, db_name):
    db_path = f"/Users/ricardo/Desktop/MECD/2ndYear/Thesis/database/{db_name}.db"
    
    try:
        con = duckdb.connect(db_path, read_only=True)
        print(f" Connected to the data warehouse: {db_name}.db")
    except Exception as e:
        print(f"Failed to connect to database: {e}")
        return
    
    try:
        with open(query_file, 'r') as f:
            query = f.read()
            query = query.strip().lower()
        ml_dataframe = con.execute(query).df()

        print(f"\n Query successful! Extracted {len(ml_dataframe)} rows and {len(ml_dataframe.columns)} features.")
        print(ml_dataframe.head())
                
        want_save = input("\nDo you want to save this dataset to a CSV file? (yes/no): ").strip().lower()
                
        if want_save == 'yes':
            out_path = f'/Users/ricardo/Desktop/MECD/2ndYear/Thesis/datasets_final/{output_name}.csv'
                    
            ml_dataframe.to_csv(
                out_path, 
                index=False,
                sep='\t',
                na_rep='NA',
                encoding='utf-8'
            )
            
            print(f" Dataset saved to: {out_path}")
            return True  
        else:
            print("Dataset discarded.")
            return False 
                
    except Exception as e:
        print(f"\n Error executing SQL query:\n{e}")
        print("Please fix the query and try again.")
        return False
    
    con.close()

    

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python3 query_db.py [query_file] [output_name] [db_name]")
        print("Example: python3 query_db.py query.sql ml_dataset data_warehouse")
        sys.exit(1)

    query_file = sys.argv[1]
    output_name = sys.argv[2]
    db_name = sys.argv[3]
    query_database(query_file=query_file, output_name=output_name, db_name=db_name)