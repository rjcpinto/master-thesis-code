import duckdb
import pandas as pd
import sys
import readline

def query_database(con, ml_query, output_name, is_show_tables):
    try:
        ml_dataframe = con.execute(ml_query).df()

        print(f"\nQuery successful! Extracted {len(ml_dataframe)} rows and {len(ml_dataframe.columns)} features.")
        if is_show_tables == "yes":
            # Print the whole thing if it's just a structural table
            print(ml_dataframe)
        else:
            # Only print the head if it's a massive ML dataset
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
            
            print(f"Dataset saved to: {out_path}")
            return True  # Returns True to tell the main loop to break
        else:
            print("Dataset discarded. Let's try writing a new query.")
            return False # Returns False so the main loop continues
                
    except Exception as e:
        print(f"\nError executing SQL query:\n{e}")
        print("Please fix the query and try again.")
        return False

def extract_ml_dataset(output_name, db_name):
    db_path = f"/Users/ricardo/Desktop/MECD/2ndYear/Thesis/database/{db_name}.db"
    
    try:
        con = duckdb.connect(db_path, read_only=True)
        print(f"Connected to the data warehouse: {db_name}.db")
    except Exception as e:
        print(f"Failed to connect to database: {e}")
        return

    while True:
        print("\n" + "="*50)
        print("Input your SQL query. You can paste multiple lines.")
        print("End your query with a semicolon ';' and press Enter.")
        print("Type 'exit' to quit.")
        print("To assess the structure of your tables, you can use 'SHOW TABLES;', 'DESCRIBE table_name;' and 'SELECT * FROM information_schema.columns;'")
        
        query_lines = []
            
        while True:
            prompt = "> " if not query_lines else "... "
            line = input(prompt)
            
            if line.strip().lower() == 'exit':
                print("Exiting...")
                con.close()
                return

            query_lines.append(line)
            
            if ';' in line:
                break
        
        ml_query = '\n'.join(query_lines)

        if not ml_query.replace(';', '').strip():
            continue

        # Correct way to check for structural queries
        ml_query_lower = ml_query.strip().lower()
        if ml_query_lower.startswith('show') or ml_query_lower.startswith('describe') or 'information_schema' in ml_query_lower:
            should_break = query_database(con, ml_query, output_name, is_show_tables="yes")
        else:
            should_break = query_database(con, ml_query, output_name, is_show_tables="no")
            
        # If query_database returned True (meaning the user saved the file), break the infinite loop
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