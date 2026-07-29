import os
import sys
import time
import joblib
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, average_precision_score
from sklearn.ensemble import RandomForestClassifier
from pulearn import BaggingPuClassifier, PUStratifiedKFold
from pathlib import Path

RESULTS_DIR = Path("results")

def get_recall_at_k(y_true, y_probs, k_percent, pos_label=1):
    k_count = max(1, int(len(y_probs) * k_percent))
    top_k_indices = np.argsort(y_probs)[::-1][:k_count]
    
    if isinstance(y_true, pd.Series):
        true_pos_in_top_k = np.sum(y_true.iloc[top_k_indices] == pos_label)
    else:
        true_pos_in_top_k = np.sum(y_true[top_k_indices] == pos_label)
        
    total_positives = np.sum(y_true == pos_label)
    return true_pos_in_top_k / total_positives if total_positives > 0 else 0.0

def custom_recall_scorer(y_true, y_probs):
    return get_recall_at_k(y_true, y_probs, k_percent=0.005)

pu_recall_scorer = make_scorer(custom_recall_scorer, needs_proba=True)

def run_tuning_and_evaluation(X_train, y_train, X_test, y_test, test_identifiers):
    print("\n" + "="*60)
    print("   Hyperparameter tuning    ")
    print("=" * 60)
    
    metadata_cols = ['User_input']
    X_train_clean = X_train.drop(columns=metadata_cols, errors='ignore')
    X_test_clean = X_test.drop(columns=metadata_cols, errors='ignore')
    
    base_rf = RandomForestClassifier(random_state=42, n_jobs=1)
    pu_model = BaggingPuClassifier(estimator=base_rf, n_estimators=50, n_jobs=-1, random_state=42)
        
    param_grid = {
        'estimator__n_estimators': [20, 50],
        'estimator__max_depth': [5, 10, 15],
        'estimator__min_samples_split': [2, 5, 10],
        'estimator__min_samples_leaf': [2, 4, 10],
        'estimator__max_features': ['sqrt', 'log2']
    }
    
    cv_strategy = PUStratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    random_search = RandomizedSearchCV(
        estimator=pu_model,
        param_distributions=param_grid,
        n_iter=15,  
        scoring=pu_recall_scorer,
        cv=cv_strategy,
        random_state=42,
        n_jobs=1,   
        verbose=2
    )
    
    print("Starting randomized search...")
    start_time = time.time()
    random_search.fit(X_train_clean, y_train)
    
    print(f"\nTuning completed in {(time.time() - start_time) / 60:.1f} minutes.")
    print("\n Best hyperparameters found:")
    for param, value in random_search.best_params_.items():
        print(f"{param.replace('estimator__', '')}: {value}")
        
    best_model = random_search.best_estimator_
    
    print("\n" + "="*60)
    print("   Testing on 20% holdout data   ")
    print("=" * 60)
    
    probs = best_model.predict_proba(X_test_clean)[:, 1]
    
    recall_01 = get_recall_at_k(y_test, probs, 0.001)
    recall_05 = get_recall_at_k(y_test, probs, 0.005)
    recall_10 = get_recall_at_k(y_test, probs, 0.010)
    pr_auc = average_precision_score(y_test, probs)
    
    print(f" Final Test Recall @ 0.1%: {recall_01:.1%}")
    print(f" Final Test Recall @ 0.5%: {recall_05:.1%}")
    print(f" Final Test Recall @ 1.0%: {recall_10:.1%}")
    print(f" Final Test PR-AUC: {pr_auc:.3f}")
    
    joblib.dump(best_model, RESULTS_DIR / "models/final_tuned_pu_rf_model.pkl")
    print(f"\nFinal model saved to: {RESULTS_DIR}/models/final_tuned_pu_rf_model.pkl")
    
    predictions_df = pd.DataFrame({
        'User_input': test_identifiers,
        'true_label': y_test,
        'predicted_prob': probs
    }).sort_values(by='predicted_prob', ascending=False)
    
    predictions_df.to_csv(RESULTS_DIR / "metrics/final_test_predictions_ranked.csv", index=False)

if __name__ == "__main__":
    os.makedirs(RESULTS_DIR / "models", exist_ok=True)
    os.makedirs(RESULTS_DIR / "metrics", exist_ok=True)

    train_path = RESULTS_DIR / "data/train_v3.csv"
    test_path = RESULTS_DIR / "data/test_v3.csv"
    
    if not train_path.exists() or not test_path.exists():
        print(f"Error: Could not find data files at {train_path} or {test_path}.")
        print("Please ensure you saved them to the 'results' directory.")
        sys.exit(1)
        
    print(f"Loading pre-built datasets:\n -> {train_path}\n -> {test_path}")
    df_train_v3 = pd.read_csv(train_path)
    df_test_v3 = pd.read_csv(test_path)
    
    y_train = df_train_v3.pop('label')
    y_test = df_test_v3.pop('label')
    test_identifiers = df_test_v3['User_input'] if 'User_input' in df_test_v3.columns else df_test_v3.index
    
    run_tuning_and_evaluation(df_train_v3, y_train, df_test_v3, y_test, test_identifiers)