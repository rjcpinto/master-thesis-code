import os
import sys
import joblib
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from kneed import KneeLocator
warnings.filterwarnings('ignore')
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, classification_report, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from pulearn import BaggingPuClassifier


"""
Different imputation strategies for handling missing values in the dataset:
v1 Baseline: Keeping the NaNs (it will be used only with the semi-supervised models)
v2 KNN Imputed: KNN fills gaps based on the 5 most similar variants
v3 MICE: A highly robust machine learning approach that models each missing feature as a function of all the other features
v4 Median Imputation
"""


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
    
    print("v1. Baseline datasets")
    datasets_train['v1'] = df_train.copy()
    datasets_test['v1'] = df_test.copy()
    
    metadata = ['label', 'User_input']
    num_cols = [c for c in df_train.select_dtypes(include=[np.number]).columns if c not in metadata]
    
    scaler = StandardScaler()

    print("v2. KNN Imputed datasets")
    X_train_scaled = scaler.fit_transform(df_train[num_cols])
    X_test_scaled = scaler.transform(df_test[num_cols])
    knn = KNNImputer(n_neighbors=5, weights='distance')
    datasets_train['v2'] = df_train.copy()
    datasets_test['v2'] = df_test.copy()
    datasets_train['v2'][num_cols] = scaler.inverse_transform(knn.fit_transform(X_train_scaled))
    datasets_test['v2'][num_cols] = scaler.inverse_transform(knn.transform(X_test_scaled))

    print("v3. MICE Imputed datasets")
    X_train_scaled_mice = scaler.fit_transform(df_train[num_cols])
    X_test_scaled_mice = scaler.transform(df_test[num_cols])
    mice = IterativeImputer(max_iter=10, random_state=42)
    datasets_train['v3'] = df_train.copy()
    datasets_test['v3'] = df_test.copy()
    datasets_train['v3'][num_cols] = scaler.inverse_transform(mice.fit_transform(X_train_scaled_mice))
    datasets_test['v3'][num_cols] = scaler.inverse_transform(mice.transform(X_test_scaled_mice))

    print("v4. Median Imputed datasets")
    median_imp = SimpleImputer(strategy='median')
    datasets_train['v4'] = df_train.copy()
    datasets_test['v4'] = df_test.copy()
    datasets_train['v4'][num_cols] = median_imp.fit_transform(datasets_train['v4'][num_cols])
    datasets_test['v4'][num_cols] = median_imp.transform(datasets_test['v4'][num_cols])

    return datasets_train, datasets_test


def run_unsupervised_analysis(train_datasets, max_k_elbow=10):
    print("\n" + "="*50)
    print("      Starting Unsupervised analysis       ")
    print("=" * 50)
    results = {}
    unsupervised_summary = []
    
    invalid_for_unsupervised = ['v1']
    
    for name, df in train_datasets.items():
        if name in invalid_for_unsupervised:
            continue
            
        print(f"\nProcessing: {name}")
        results[name] = {}
        
        metadata_cols = ['label', 'User_input']
        X = df.drop(columns=metadata_cols, errors='ignore')
        saved_indices = X.index
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        
        ## PCA analysis
        print("\n Performing PCA...")
        pca = PCA(n_components=0.95, random_state=42)
        X_pca = pca.fit_transform(X_scaled)
        
        n_pca_components = pca.n_components_
        explained_variance = sum(pca.explained_variance_ratio_)
        columns_labels = [f'PC{i+1}' for i in range(n_pca_components)]
        
        loadings = pd.DataFrame(pca.components_.T, columns=columns_labels, index=X.columns)
        loadings.to_csv(f"results/metrics/pca_loadings_{name}.csv")

        plt.figure(figsize=(8, 12)) 
        sns.heatmap(loadings, annot=True, cmap='RdBu_r', center=0, fmt='.2f')
        plt.savefig(f"results/plots/pca_loadings_{name}.svg")
        plt.savefig(f"results/plots/pca_loadings_{name}.png", dpi=300)
        plt.close()
        
        
        ## K-Means analysis
        print("\n Performing K-Means...")
        wcss = []
        for k in range(1, max_k_elbow):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X_scaled)
            wcss.append(kmeans.inertia_)

        plt.plot(range(1, max_k_elbow), wcss, marker='o')
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('WCSS')
        plt.savefig(f"results/plots/elbow_{name}.svg")
        plt.savefig(f"results/plots/elbow_{name}.png", dpi=300)
        plt.close()
        
        kl = KneeLocator(range(1, max_k_elbow), wcss, curve="convex", direction="decreasing")
        optimal_k = kl.elbow if kl.elbow else 2
        
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_pca)
        sil_score_kmeans = silhouette_score(X_pca, cluster_labels)
        
        results[name]['kmeans'] = {'silhouette_score': sil_score_kmeans, 'variance_retained': explained_variance}
        
        pca_viz = pd.DataFrame(data=X_pca[:, :2], columns=['PC1', 'PC2'])
        pca_viz['Cluster'] = cluster_labels
        
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x='PC1', 
                        y='PC2', 
                        hue='Cluster', 
                        data=pca_viz,
                        palette='viridis', 
                        s=60, alpha=0.7)
        plt.savefig(f"results/plots/pca_scatter_{name}.svg")
        plt.savefig(f"results/plots/pca_scatter_{name}.png", dpi=300)
        plt.close()
        
        centroids = pd.DataFrame(kmeans.cluster_centers_, columns=X.columns)
        centroids.to_csv(f"results/metrics/kmeans_centroids_{name}.csv")
        
        if X_scaled.shape[0] > 50000:
            print("\n -> Subsampling to 50,000 rows for DBSCAN & t-SNE to save memory...")
            np.random.seed(42)
            sub_idx = np.random.choice(X_scaled.shape[0], 50000, replace=False)
            X_run = X_scaled[sub_idx]
            cluster_labels_run = cluster_labels[sub_idx]
            run_indices = saved_indices[sub_idx]
        else:
            X_run = X_scaled
            cluster_labels_run = cluster_labels
            run_indices = saved_indices
        
        
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

        plt.figure(figsize=(10, 6))
        plt.plot(sorted_distances)
        if kneedle.knee:
            plt.axhline(y=optimal_eps, 
                        color='r', 
                        linestyle='--', 
                        label=f'Optimal eps: {optimal_eps:.2f}')
            plt.axvline(x=kneedle.knee, 
                        color='r', 
                        linestyle='--')
            plt.legend()
        plt.ylabel('Epsilon (eps) distance')
        plt.savefig(f"results/plots/dbscan_k_distance_{name}.svg")
        plt.savefig(f"results/plots/dbscan_k_distance_{name}.png", dpi=300)
        plt.close()
        
        dbscan = DBSCAN(eps=optimal_eps, min_samples=40)
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
        
        
        ## t-SNE
        print("\n Performing t-SNE...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        tsne_results = tsne.fit_transform(X_run)
        
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x=tsne_results[:, 0], 
                        y=tsne_results[:, 1], 
                        hue=cluster_labels_run, 
                        palette='Set1', 
                        s=60, alpha=0.7)
        plt.savefig(f"results/plots/tsne_clusters_{name}.svg")
        plt.savefig(f"results/plots/tsne_clusters_{name}.png", dpi=300)
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
            
        labels_df.to_csv(f"results/data/unsupervised_labels_and_tsne_{name}.csv")

    summary_df = pd.DataFrame(unsupervised_summary)
    summary_df.to_csv("results/metrics/unsupervised_summary_metrics.csv", index=False)
    
    return results


def run_pu_learning_analysis(train_datasets, test_datasets):
    print("\n" + "="*50)
    print("      Starting PU analysis       ")
    print("=" * 50)
    results = {}
    pu_summary = []
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for name in train_datasets.keys():
        print(f"\n Processing: {name}")
        results[name] = {}
        
        X_train = train_datasets[name].copy()
        X_test = test_datasets[name].copy()
        
        y_train = X_train.pop('label')
        y_test = X_test.pop('label')
        
        test_identifiers = X_test['User_input'] if 'User_input' in X_test.columns else X_test.index
        
        metadata_cols = ['User_input']
        X_train = X_train.drop(columns=metadata_cols, errors='ignore')
        X_test = X_test.drop(columns=metadata_cols, errors='ignore')
        
        print(" Running 5-Fold stratified cross-validation...")
        cv_auprc_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            X_cv_train, X_cv_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            base_estimator = HistGradientBoostingClassifier(random_state=42)
            cv_model = BaggingPuClassifier(estimator=base_estimator, n_estimators=50, n_jobs=-1, random_state=42)
            cv_model.fit(X_cv_train, y_cv_train)
            
            probs_cv = cv_model.predict_proba(X_cv_val)[:, 1]
            cv_auprc_scores.append(average_precision_score(y_cv_val, probs_cv))
            
        mean_cv_auprc = np.mean(cv_auprc_scores)
        std_cv_auprc = np.std(cv_auprc_scores)
            
        print(" Training final model on entire 80% Train set...")
        base_estimator = HistGradientBoostingClassifier(random_state=42)
        final_pu_model = BaggingPuClassifier(estimator=base_estimator, n_estimators=50, n_jobs=-1, random_state=42)
        
        final_pu_model.fit(X_train, y_train)
        
        probs = final_pu_model.predict_proba(X_test)[:, 1]
        preds = final_pu_model.predict(X_test)
        test_auprc = average_precision_score(y_test, probs)
        
        report = classification_report(y_test, preds, target_names=['Unlabeled (-1)', 'Positive (1)'])
        print(report)
                
        joblib.dump(final_pu_model, f"results/models/pu_model_{name}.pkl")
        
        with open(f"results/metrics/pu_classification_report_{name}.txt", "w") as f:
            f.write(report)
            
        pu_summary.append({
            'strategy': name,
            'cv_mean_auprc': mean_cv_auprc,
            'cv_std_auprc': std_cv_auprc,
            'test_auprc': test_auprc
        })
        
        predictions_df = pd.DataFrame({
            'User_input': test_identifiers,
            'true_label': y_test,
            'predicted_prob': probs,
            'predicted_label': preds
        })
        predictions_df.to_csv(f"results/data/pu_test_predictions_{name}.csv", index=False)
        
        results[name]['pu_learning'] = final_pu_model

    pu_summary_df = pd.DataFrame(pu_summary)
    pu_summary_df.to_csv("results/metrics/pu_summary_metrics.csv", index=False)

    return results


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 ml_pipeline.py <input_file>")
        sys.exit(1)
        
    os.makedirs("results/plots", exist_ok=True)
    os.makedirs("results/models", exist_ok=True)
    os.makedirs("results/metrics", exist_ok=True)
    os.makedirs("results/data", exist_ok=True)

    combined_df = pd.read_csv(sys.argv[1])
    
    df_train, df_test = create_stratified_splits(combined_df, test_size=0.20)
    train_sets, test_sets = generate_imputed_datasets(df_train, df_test)
    
    unsupervised_results = run_unsupervised_analysis(train_sets, max_k_elbow=10)
    supervised_results = run_pu_learning_analysis(train_sets, test_sets)