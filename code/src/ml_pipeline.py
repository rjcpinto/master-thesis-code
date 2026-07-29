import os
import sys
import time
import joblib
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from kneed import KneeLocator
warnings.filterwarnings('ignore')
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from pulearn import BaggingPuClassifier, PUStratifiedKFold
from pathlib import Path
import shap

"""
Different imputation strategies for handling missing values in the dataset:
v1 KNN Imputed: KNN fills gaps based on the 5 most similar variants
v2 MICE: A highly robust machine learning approach that models each missing feature as a function of all the other features
v3 Median Imputation
"""

RESULTS_DIR = Path("results")

def create_stratified_splits(combined_df, test_size=0.20):
    print(f"\nCreating stratified splits (Train: {1-test_size:.2f}, Test: {test_size})")
    
    df_train, df_test = train_test_split(
        combined_df, 
        test_size=test_size, 
        random_state=42, 
        stratify=combined_df['label']
    )
    
    pos_train = (df_train['label'] == 1).sum()
    pos_test = (df_test['label'] == 1).sum()
    
    print(f"Train size: {len(df_train)} (Positives: {pos_train})")
    print(f"Test size: {len(df_test)} (Positives: {pos_test})")
    
    return df_train, df_test


def generate_imputed_datasets(df_train, df_test):
    print("\nGenerating imputation strategies...")
    datasets_train = {}
    datasets_test = {}
    
    metadata = ['label', 'User_input']
    num_cols = [c for c in df_train.select_dtypes(include=[np.number]).columns if c not in metadata]
    
    scaler = StandardScaler()

    print("v1. KNN Imputed datasets")
    X_train_scaled = scaler.fit_transform(df_train[num_cols])
    X_test_scaled = scaler.transform(df_test[num_cols])
    knn = KNNImputer(n_neighbors=5, weights='distance')
    datasets_train['v1'] = df_train.copy()
    datasets_test['v1'] = df_test.copy()
    datasets_train['v1'][num_cols] = scaler.inverse_transform(knn.fit_transform(X_train_scaled))
    datasets_test['v1'][num_cols] = scaler.inverse_transform(knn.transform(X_test_scaled))

    print("v2. MICE Imputed datasets")
    X_train_scaled_mice = scaler.fit_transform(df_train[num_cols])
    X_test_scaled_mice = scaler.transform(df_test[num_cols])
    mice = IterativeImputer(max_iter=20, random_state=42)
    datasets_train['v2'] = df_train.copy()
    datasets_test['v2'] = df_test.copy()
    datasets_train['v2'][num_cols] = scaler.inverse_transform(mice.fit_transform(X_train_scaled_mice))
    datasets_test['v2'][num_cols] = scaler.inverse_transform(mice.transform(X_test_scaled_mice))

    print("v3. Median Imputed datasets")
    median_imp = SimpleImputer(strategy='median')
    datasets_train['v3'] = df_train.copy()
    datasets_test['v3'] = df_test.copy()
    datasets_train['v3'][num_cols] = median_imp.fit_transform(datasets_train['v3'][num_cols])
    datasets_test['v3'][num_cols] = median_imp.transform(datasets_test['v3'][num_cols])
    joblib.dump(median_imp, RESULTS_DIR / "models/median_imputer.pkl")

    print("\nSaving generated datasets to disk...")
    for version in datasets_train.keys():
        datasets_train[version].to_csv(RESULTS_DIR / f"data/train_{version}.csv", index=False)
        datasets_test[version].to_csv(RESULTS_DIR / f"data/test_{version}.csv", index=False)

    return datasets_train, datasets_test


def run_unsupervised_analysis(train_datasets, max_k_elbow=10):
    print("\n" + "="*50)
    print("      Starting Unsupervised analysis       ")
    print("=" * 50)
    results = {}
    unsupervised_summary = []
        
    for name, df in train_datasets.items():
            
        print(f"\nProcessing: {name}")
        results[name] = {}
        
        metadata_cols = ['label', 'User_input']
        X = df.drop(columns=metadata_cols, errors='ignore')
        saved_indices = X.index
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        results[name]['scaler'] = scaler
        
        ## PCA analysis
        print("\n Performing PCA...")
        pca = PCA(n_components=0.95, random_state=42)
        X_pca = pca.fit_transform(X_scaled)
        
        n_pca_components = pca.n_components_
        explained_variance = sum(pca.explained_variance_ratio_)
        columns_labels = [f'PC{i+1}' for i in range(n_pca_components)]
        
        pc1_var = pca.explained_variance_ratio_[0] * 100
        pc2_var = pca.explained_variance_ratio_[1] * 100
        
        variance_df = pd.DataFrame({
            'PC': columns_labels,
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'explained_variance_percent': pca.explained_variance_ratio_ * 100,
            'cumulative_variance_percent': np.cumsum(pca.explained_variance_ratio_) * 100
        })
        variance_df.to_csv(RESULTS_DIR / f"metrics/pca_variance_{name}.csv", index=False)
        
        loadings = pd.DataFrame(pca.components_.T, columns=columns_labels, index=X.columns)
        loadings.to_csv(RESULTS_DIR / f"metrics/pca_loadings_{name}.csv")

        plt.figure(figsize=(8, 12), facecolor='white') 
        sns.heatmap(loadings, annot=False, cmap='seismic', center=0, fmt='.2f')
        plt.tight_layout() 
        plt.savefig(RESULTS_DIR / f"plots/pca_loadings_{name}.svg", bbox_inches='tight')
        plt.savefig(RESULTS_DIR / f"plots/pca_loadings_{name}.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        df_pc = pd.DataFrame(X_pca, columns=columns_labels)
        df_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        
        correlations = []
        for col in df_scaled.columns:
            corr_series = df_pc.corrwith(df_scaled[col])
            corr_series.name = col
            correlations.append(corr_series)
            
        corr_df = pd.DataFrame(correlations)
        corr_df.to_csv(RESULTS_DIR / f"metrics/pca_feature_correlations_{name}.csv")
        
        
        ## K-Means analysis
        print("\n Performing K-Means...")
        wcss = []
        for k in range(1, max_k_elbow):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X_pca) 
            wcss.append(kmeans.inertia_)

        plt.figure(figsize=(10, 6), facecolor='white')
        plt.plot(range(1, max_k_elbow), wcss, marker='o', color=sns.color_palette("tab10")[0])
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('WCSS')
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f"plots/elbow_{name}.svg", bbox_inches='tight')
        plt.savefig(RESULTS_DIR / f"plots/elbow_{name}.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        kl = KneeLocator(range(1, max_k_elbow), wcss, curve="convex", direction="decreasing")
        optimal_k = kl.elbow if kl.elbow else 2
        
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_pca)
        sil_score_kmeans = silhouette_score(X_pca, cluster_labels)
        
        # Random Forest feature importance for K-Means clusters
        print(" Extracting feature importances for K-Means clusters using Random Forest...")
        rf_cluster_explainer = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf_cluster_explainer.fit(X_scaled, cluster_labels)
        
        rf_importances = pd.Series(
            rf_cluster_explainer.feature_importances_, 
            index=X.columns
        ).sort_values(ascending=False)
        
        rf_importances.to_csv(RESULTS_DIR / f"metrics/kmeans_rf_importances_{name}.csv")
        
        plt.figure(figsize=(10, 8), facecolor='white')
        sns.barplot(x=rf_importances.head(20).values, y=rf_importances.head(20).index, color=sns.color_palette("tab10")[0])
        plt.xlabel('Random Forest Feature Importance')
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f"plots/kmeans_rf_importance_{name}.svg", bbox_inches='tight')
        plt.savefig(RESULTS_DIR / f"plots/kmeans_rf_importance_{name}.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        results[name]['pca_model'] = pca
        results[name]['kmeans'] = {
            'model': kmeans,
            'silhouette_score': sil_score_kmeans, 
            'variance_retained': explained_variance,
            'rf_explainer_model': rf_cluster_explainer 
        }
        
        pca_viz = pd.DataFrame(data=X_pca[:, :2], columns=['PC1', 'PC2'])
        pca_viz['Cluster'] = cluster_labels
        
        plt.figure(figsize=(10, 8), facecolor='white')
        sns.scatterplot(x='PC1', 
                        y='PC2', 
                        hue='Cluster', 
                        data=pca_viz,
                        palette='Dark2', 
                        s=60, alpha=0.7)
        plt.xlabel(f'PC1 ({pc1_var:.1f}%)')
        plt.ylabel(f'PC2 ({pc2_var:.1f}%)')
        plt.savefig(RESULTS_DIR / f"plots/pca_kmeans_{name}.svg", bbox_inches='tight')
        plt.savefig(RESULTS_DIR / f"plots/pca_kmeans_{name}.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        centroids_original_space = pca.inverse_transform(kmeans.cluster_centers_)
        centroids = pd.DataFrame(centroids_original_space, columns=X.columns)
        centroids.to_csv(RESULTS_DIR / f"metrics/kmeans_centroids_{name}.csv")
        
        
        
        if X_scaled.shape[0] > 50000:
            print("\n Subsampling to 50,000 rows for DBSCAN & t-SNE to save memory...")
            
            if 'label' in df.columns:
                y = df['label'].values
                pos_idx = np.where(y == 1)[0]
                unl_idx = np.where(y != 1)[0]
                
                n_pos_to_keep = min(len(pos_idx), 50000)
                n_unl_to_keep = 50000 - n_pos_to_keep
                
                np.random.seed(42)
                sampled_pos = np.random.choice(pos_idx, n_pos_to_keep, replace=False)
                sampled_unl = np.random.choice(unl_idx, n_unl_to_keep, replace=False)
                
                sub_idx = np.concatenate([sampled_pos, sampled_unl])
                np.random.shuffle(sub_idx) 
            else:
                np.random.seed(42)
                sub_idx = np.random.choice(X_scaled.shape[0], 50000, replace=False)
                
            X_run = X_scaled[sub_idx]
            cluster_labels_run = cluster_labels[sub_idx]
            run_indices = saved_indices[sub_idx]
            X_pca_run = X_pca[sub_idx]
        else:
            X_run = X_scaled
            cluster_labels_run = cluster_labels
            run_indices = saved_indices
            X_pca_run = X_pca
        
        
        ## DBSCAN
        print("\n Performing DBSCAN...")
        k = 20
        nbrs = NearestNeighbors(n_neighbors=k).fit(X_run)
        distances, indices = nbrs.kneighbors(X_run)
        sorted_distances = np.sort(distances[:, k-1], axis=0)

        x_points = np.arange(len(sorted_distances))
        kneedle = KneeLocator(x_points, 
                              sorted_distances, 
                              curve="convex", 
                              direction="increasing")
        optimal_eps = sorted_distances[kneedle.knee] if kneedle.knee else 4.0

        plt.figure(figsize=(10, 6), facecolor='white')
        plt.plot(sorted_distances, color=sns.color_palette("tab10")[0])
        if kneedle.knee:
            plt.axhline(y=optimal_eps, 
                        color=sns.color_palette("tab10")[1], 
                        linestyle='--', 
                        label=f'Optimal eps: {optimal_eps:.2f}')
            plt.axvline(x=kneedle.knee, 
                        color=sns.color_palette("tab10")[1], 
                        linestyle='--')
            plt.legend()
        plt.ylabel('Epsilon (eps) distance')
        plt.tight_layout() 
        plt.savefig(RESULTS_DIR / f"plots/dbscan_k_distance_{name}.svg", bbox_inches='tight')
        plt.savefig(RESULTS_DIR / f"plots/dbscan_k_distance_{name}.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        dbscan = DBSCAN(eps=optimal_eps, min_samples=40, n_jobs=-1)
        clusters_dbscan = dbscan.fit_predict(X_run)
        
        if len(set(clusters_dbscan)) > 1:
            mask = clusters_dbscan != -1
            if mask.sum() > 0 and len(set(clusters_dbscan[mask])) > 1:
                sil_score_dbscan = silhouette_score(X_run[mask], clusters_dbscan[mask])
            else:
                sil_score_dbscan = -1
        else:
            sil_score_dbscan = -1
        
        results[name]['dbscan'] = {'silhouette_score': sil_score_dbscan}
        

        ## Random Forest feature importance for DBSCAN
        if len(set(clusters_dbscan)) > 1:
            print(" Extracting feature importances for DBSCAN clusters using Random Forest...")
            rf_dbscan_explainer = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            rf_dbscan_explainer.fit(X_run, clusters_dbscan)
            
            rf_dbscan_importances = pd.Series(
                rf_dbscan_explainer.feature_importances_, 
                index=X.columns
            ).sort_values(ascending=False)
            
            rf_dbscan_importances.to_csv(RESULTS_DIR / f"metrics/dbscan_rf_importances_{name}.csv")
            
            plt.figure(figsize=(10, 8), facecolor='white')
            sns.barplot(x=rf_dbscan_importances.head(20).values, y=rf_dbscan_importances.head(20).index, color=sns.color_palette("tab10")[0])
            plt.xlabel('RF feature importance')
            plt.tight_layout()
            plt.savefig(RESULTS_DIR / f"plots/dbscan_rf_importance_{name}.svg", bbox_inches='tight')
            plt.savefig(RESULTS_DIR / f"plots/dbscan_rf_importance_{name}.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            results[name]['dbscan']['rf_explainer_model'] = rf_dbscan_explainer
        else:
            print(" Skipping DBSCAN Random Forest (no clusters found).")
            
        
        ## t-SNE
        print("\n Performing t-SNE...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        
        tsne_results = tsne.fit_transform(X_run)
        
        plt.figure(figsize=(10, 8), facecolor='white')
        sns.scatterplot(x=tsne_results[:, 0], 
                        y=tsne_results[:, 1],                         
                        hue=cluster_labels_run, 
                        palette='Dark2', 
                        s=60, alpha=0.7)
        plt.savefig(RESULTS_DIR / f"plots/tsne_kmeans_{name}.svg", bbox_inches='tight')
        plt.savefig(RESULTS_DIR / f"plots/tsne_kmeans_{name}.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        plt.figure(figsize=(10, 8), facecolor='white')
        unique_dbscan_labels = sorted(list(set(clusters_dbscan)))
        n_clusters = len([l for l in unique_dbscan_labels if l != -1])
        dbscan_palette = sns.color_palette("Dark2", n_clusters)
        if -1 in unique_dbscan_labels:
            dbscan_palette.insert(0, (0.7, 0.7, 0.7)) 
            
        sns.scatterplot(x=tsne_results[:, 0], 
                        y=tsne_results[:, 1],                         
                        hue=clusters_dbscan, 
                        palette=dbscan_palette, 
                        s=60, alpha=0.7)
        plt.savefig(RESULTS_DIR / f"plots/tsne_dbscan_{name}.svg", bbox_inches='tight')
        plt.savefig(RESULTS_DIR / f"plots/tsne_dbscan_{name}.png", dpi=300, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(10, 8), facecolor='white')
        sns.scatterplot(x=X_pca_run[:, 0], 
                        y=X_pca_run[:, 1],                         
                        hue=clusters_dbscan, 
                        palette=dbscan_palette, 
                        s=60, alpha=0.7)
        plt.xlabel(f'PC1 ({pc1_var:.1f}%)')
        plt.ylabel(f'PC2 ({pc2_var:.1f}%)')
        plt.savefig(RESULTS_DIR / f"plots/pca_dbscan_{name}.svg", bbox_inches='tight')
        plt.savefig(RESULTS_DIR / f"plots/pca_dbscan_{name}.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        unsupervised_summary.append({
            'strategy': name,
            'pca_retained_var': explained_variance,
            'optimal_k': optimal_k,
            'kmeans_silhouette': sil_score_kmeans,
            'optimal_eps': optimal_eps,
            'dbscan_silhouette': sil_score_dbscan
        })
        
        labels_df = pd.DataFrame({
            
            'kmeans_cluster': cluster_labels_run,              
            'dbscan_cluster': clusters_dbscan,
            'tsne_1': tsne_results[:, 0],
            'tsne_2': tsne_results[:, 1]            
        }, index=run_indices) 
        
        if 'User_input' in df.columns:
            labels_df = labels_df.join(df['User_input'])
            
        labels_df.to_csv(RESULTS_DIR / f"metrics/unsupervised_labels_and_tsne_{name}.csv")

    summary_df = pd.DataFrame(unsupervised_summary)
    summary_df.to_csv(RESULTS_DIR / f"metrics/unsupervised_summary_metrics.csv", index=False)
    
    joblib.dump(results, RESULTS_DIR / "models/unsupervised_results.pkl")
    
    return results



def generate_all_cluster_profiles(train_datasets, unsupervised_results, top_n=15):
    print("\n" + "="*50)
    print("   Generating SHAP profiles for K-Means & DBSCAN   ")
    print("=" * 50)
    
    for name, df in train_datasets.items():
        print(f"\nProfiling clusters for: {name}")
        
        labels_path = RESULTS_DIR / f"metrics/unsupervised_labels_and_tsne_{name}.csv"
        if not labels_path.exists():
            print(f" Skipping {name}: labels file not found.")
            continue
            
        labels_df = pd.read_csv(labels_path, index_col=0)
        
        metadata_cols = ['label', 'User_input']
        features_df = df.drop(columns=[c for c in metadata_cols if c in df.columns])
        
        features_aligned = features_df.loc[labels_df.index]
        
        scaler = unsupervised_results[name]['scaler']
        X_scaled = pd.DataFrame(
            scaler.transform(features_aligned), 
            columns=features_aligned.columns, 
            index=features_aligned.index
        )
        
        algorithms = ['kmeans', 'dbscan']
        
        for algo in algorithms:
            if algo not in unsupervised_results[name] or 'rf_explainer_model' not in unsupervised_results[name][algo]:
                print(f" Skipping {algo.upper()}: RF model not found in unsupervised_results.")
                continue
                
            print(f" -> Extracting SHAP directions and means for {algo.upper()}...")
            rf_model = unsupervised_results[name][algo]['rf_explainer_model']
            
            X_shap_sample = shap.sample(X_scaled, min(5000, len(X_scaled)), random_state=42)
            
            explainer = shap.TreeExplainer(rf_model)
            shap_values = explainer.shap_values(X_shap_sample)
            
            is_list = isinstance(shap_values, list)
            
            rf_classes = list(rf_model.classes_) 
            
            all_cluster_stats = []
            
            for idx, c in enumerate(rf_classes):                    
                if is_list:
                    class_shap_vals = shap_values[idx]
                else:
                    class_shap_vals = shap_values[:, :, idx] 
                
                cluster_mask = (labels_df[f"{algo}_cluster"] == c)
                
                for col_idx, col_name in enumerate(X_scaled.columns):
                    importance = np.abs(class_shap_vals[:, col_idx]).mean()
                    
                    feature_vals = X_shap_sample.iloc[:, col_idx].values
                    s_vals = class_shap_vals[:, col_idx]
                    
                    if np.std(feature_vals) == 0 or np.std(s_vals) == 0:
                        corr = 0.0
                    else:
                        corr = np.corrcoef(feature_vals, s_vals)[0, 1]
                    
                    if np.isnan(corr) or importance == 0:
                        direction = "Neutral/Zero Variance"
                    elif corr > 0.3:
                        direction = "High values drive cluster"
                    elif corr < -0.3:
                        direction = "Low values drive cluster"
                    else:
                        direction = "Complex/Non-linear"
                        
                    raw_cluster_mean = features_aligned.loc[cluster_mask, col_name].mean()
                    raw_global_mean = features_aligned[col_name].mean()
                        
                    all_cluster_stats.append({
                        'Cluster': c,
                        'Feature': col_name,
                        'SHAP_Importance': importance,
                        'Direction': direction,
                        'Cluster_Mean': raw_cluster_mean,
                        'Global_Mean': raw_global_mean,
                        'Correlation': corr
                    })
            
            if not all_cluster_stats:
                continue
                
            stats_df = pd.DataFrame(all_cluster_stats)
            
            for c in rf_classes:
                if c == -1: continue
                
                c_df = stats_df[stats_df['Cluster'] == c].sort_values(by='SHAP_Importance', ascending=False)
                
                c_df_top = c_df.head(top_n)
                
                save_path = RESULTS_DIR / f"metrics/{algo}_cluster_{c}_detailed_profile_{name}.csv"
                c_df_top.to_csv(save_path, index=False)
                
            print(f"    Saved rich SHAP profiles to metrics/{algo}_cluster_[X]_detailed_profile_{name}.csv")
            
            

def get_recall_at_k(y_true, y_probs, k_percent, pos_label=1):
    k_count = max(1, int(len(y_probs) * k_percent))
    
    top_k_indices = np.argsort(y_probs)[::-1][:k_count]
    
    if isinstance(y_true, pd.Series):
        true_pos_in_top_k = np.sum(y_true.iloc[top_k_indices] == pos_label)
    else:
        true_pos_in_top_k = np.sum(y_true[top_k_indices] == pos_label)
        
    total_positives = np.sum(y_true == pos_label)
    
    if total_positives == 0:
        return 0.0
    return true_pos_in_top_k / total_positives



def run_comprehensive_tournament(all_train_datasets):
    print("\n" + "="*50)
    print("  Base estimator & Dataset tournament   ")
    print("=" * 50)
    
    estimators_to_test = {
    'LogisticRegression': LogisticRegression(random_state=42, max_iter=2000, n_jobs=1),
    'HistGradientBoosting': HistGradientBoostingClassifier(random_state=42),
    'LightGBM': LGBMClassifier(random_state=42, n_jobs=1), 
    'XGBoost': XGBClassifier(random_state=42, n_jobs=1, eval_metric='logloss'),
    'CatBoost': CatBoostClassifier(random_state=42, thread_count=1, verbose=0),    
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1),
    'ExtraTrees': ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=1)
}
    
    cv_strategy = PUStratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    tournament_results = []
    
    for dataset_name, df_train in all_train_datasets.items():
        print(f"\n[ Evaluating dataset: {dataset_name} ]")
        
        X_train = df_train.copy()
        y_train = X_train.pop('label')
        
        metadata_cols = ['User_input']
        X_train = X_train.drop(columns=metadata_cols, errors='ignore')
        
        for est_name, base_est in estimators_to_test.items():
            print(f" Testing {est_name}...", end=" ")
            start_time = time.time()
            
            pu_model = BaggingPuClassifier(estimator=base_est, n_estimators=50, n_jobs=-1, random_state=42)
            
            fold_scores_01 = []
            fold_scores_05 = []
            fold_scores_10 = []
            fold_scores_pr_auc = [] 
            
            for train_idx, val_idx in cv_strategy.split(X_train, y_train):
                X_cv_train, X_cv_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                
                pu_model.fit(X_cv_train, y_cv_train)
                probs_cv = pu_model.predict_proba(X_cv_val)[:, 1]
                
                fold_scores_01.append(get_recall_at_k(y_cv_val, probs_cv, k_percent=0.001))
                fold_scores_05.append(get_recall_at_k(y_cv_val, probs_cv, k_percent=0.005))
                fold_scores_10.append(get_recall_at_k(y_cv_val, probs_cv, k_percent=0.010))
                fold_scores_pr_auc.append(average_precision_score(y_cv_val, probs_cv)) 
                
            mean_score_01 = np.mean(fold_scores_01)
            mean_score_05 = np.mean(fold_scores_05)
            mean_score_10 = np.mean(fold_scores_10)
            mean_pr_auc = np.mean(fold_scores_pr_auc)
            exec_time = time.time() - start_time
            
            print(f"R@0.1%: {mean_score_01:.1%} | R@0.5%: {mean_score_05:.1%} | R@1.0%: {mean_score_10:.1%} | PR-AUC: {mean_pr_auc:.3f} ({exec_time:.1f}s)")
            
            tournament_results.append({
                'dataset': dataset_name,
                'estimator': est_name,
                'recall_at_01_pct': mean_score_01,
                'recall_at_05_pct': mean_score_05,
                'recall_at_10_pct': mean_score_10,
                'pr_auc': mean_pr_auc, 
                'execution_time_sec': round(exec_time, 1)
            })
            
    results_df = pd.DataFrame(tournament_results)
    results_df = results_df.sort_values(by='recall_at_05_pct', ascending=False)
    
    print("\n" + "="*50)
    print("   Tournament complete - Top 5 combinations   ")
    print("=" * 50)
    print(results_df.head(5).to_string(index=False))
    
    results_df.to_csv(RESULTS_DIR / "metrics/master_tournament_rankings.csv", index=False)
    print(f"\nFull tournament rankings saved to: {RESULTS_DIR}/metrics/master_tournament_rankings.csv")
    
    return results_df


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 ml_pipeline.py <input_file>")
        sys.exit(1)
        
    os.makedirs(RESULTS_DIR / "plots", exist_ok=True)
    os.makedirs(RESULTS_DIR / "models", exist_ok=True)
    os.makedirs(RESULTS_DIR / "metrics", exist_ok=True)
    os.makedirs(RESULTS_DIR / "data", exist_ok=True)

    combined_df = pd.read_csv(sys.argv[1])
    
    df_train, df_test = create_stratified_splits(combined_df, test_size=0.20)
    train_sets, test_sets = generate_imputed_datasets(df_train, df_test)
    
    unsupervised_results = run_unsupervised_analysis(train_sets, max_k_elbow=10)
    
    generate_all_cluster_profiles(train_sets, unsupervised_results, top_n=15) 
    
    all_training_datasets = {**train_sets}
    
    tournament_results_df = run_comprehensive_tournament(all_training_datasets)