import sys
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

RESULTS_DIR = Path("results")

def run_discovery(input_file):
    print("\n" + "="*60)
    print("   Starting variant discovery...   ")
    print("=" * 60)
    
    df = pd.read_csv(input_file)
    
    unlabeled_df = df[df['label'] != 1].copy()
        
    identifiers = unlabeled_df['User_input'] if 'User_input' in unlabeled_df.columns else unlabeled_df.index
    
    metadata_cols = ['label', 'User_input']
    num_cols = [c for c in unlabeled_df.select_dtypes(include=[np.number]).columns if c not in metadata_cols]
    
    X_unlabeled = unlabeled_df[num_cols].copy()
    
    print("Applying Median imputation to missing values...")
    imputer = joblib.load(RESULTS_DIR / "models/median_imputer.pkl")
    X_unlabeled_imputed = imputer.fit_transform(X_unlabeled)
    
    X_unlabeled = pd.DataFrame(X_unlabeled_imputed, columns=num_cols, index=X_unlabeled.index)
    
    model_path = RESULTS_DIR / "models/final_tuned_pu_rf_model.pkl"
    if not model_path.exists():
        print(f"\nError: Could not find the tuned model at {model_path}.")
        sys.exit(1)
        
    print(f"Loading tuned model: {model_path.name}")
    model = joblib.load(model_path)
    
    print("Calculating probabilities for all variants...")
    probs = model.predict_proba(X_unlabeled)[:, 1]
    
    results_df = pd.DataFrame({
        'User_input': identifiers,
        'predicted_probability': probs
    }).sort_values(by='predicted_probability', ascending=False)
    
    output_path = RESULTS_DIR / "metrics/top_putative_positives.csv"
    results_df.to_csv(output_path, index=False)
    
    print("\n" + "="*60)
    print("   Discovery complete!   ")
    print("=" * 60)
    print(f"Full ranked list saved to: {output_path}\n")
    
    print("Top 10 putative Positives:")
    print(results_df.head(10).to_string(index=False))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 discover_positives.py <raw_combined_dataset.csv>")
        sys.exit(1)
        
    run_discovery(sys.argv[1])