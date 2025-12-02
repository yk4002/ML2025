#USE THIS AS THE THING TO RUN OVERNIGHT

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_mutual_info_score, adjusted_rand_score
from sklearn.model_selection import KFold
from tqdm import tqdm


# block 1
# Load the MNIST dataset
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist["data"], mnist["target"].astype(int)


# block 2
# Visualize sampled images and labels as a grid plot with one row per class and five images per class
f=(15,10)
classes = 10 # classes from 0-9
imgs_per_class = 5
fig, axes = plt.subplots(classes, imgs_per_class, figsize=f)
for mnist_dig in range(classes):
    indexes = np.where(y==mnist_dig)[0] # get all indexes for the current digit
    chosen = np.random.choice(indexes, imgs_per_class, replace=False) # randomly pick images
    for i, idx in enumerate(chosen):
        ax = axes[mnist_dig, i]
        ax.imshow(X[idx].reshape(28, 28), cmap="gray") # reshape to MNIST image
        ax.axis("off")
plt.suptitle("Sampled MNIST images with class labels", fontsize=20)
plt.show()


# block 3
# Dimensionality reduction to 100 features so that training is faster
n = 100
scaler = StandardScaler(with_std=False)
X_centred = scaler.fit_transform(X)
pca = PCA(n_components=n)
X_PCA = pca.fit_transform(X_centred)
print("PCA completed, reduced to 100 features.\n")


# block 5
# GMM with K-Fold CV
hyperparams_gmm = {
    "n_components": [10],
    "covariance_type": ["full", "diag"],
    "n_init": [5, 10]
}

cvfunc = KFold(n_splits=3, shuffle=True, random_state=42)
total_g_runs = len(hyperparams_gmm["n_components"]) * len(hyperparams_gmm["covariance_type"]) * len(hyperparams_gmm["n_init"])
pbar = tqdm(total=total_g_runs, desc="GMM K-Fold CV")
best_score_g = -1
best_params_g = None

for n_comp in hyperparams_gmm["n_components"]:
    for cov in hyperparams_gmm["covariance_type"]:
        for n_init in hyperparams_gmm["n_init"]:
            fold_scores = []
            for train_idx, val_idx in cvfunc.split(X_PCA):
                X_train, X_val = X_PCA[train_idx], X_PCA[val_idx]
                gmm = GaussianMixture(
                    n_components=n_comp, 
                    covariance_type=cov, 
                    n_init=n_init, 
                    random_state=42
                )
                gmm.fit(X_train)
                labels_val = gmm.predict(X_val)
                sil = silhouette_score(X_val, labels_val)
                fold_scores.append(sil)

            avg_sil = np.mean(fold_scores)
            if avg_sil > best_score_g:
                best_score_g = avg_sil
                best_params_g = {"n_components": n_comp,"covariance_type": cov,"n_init": n_init}
            pbar.update(1)
pbar.close()
print("Best GMM params:", best_params_g)

gmm = GaussianMixture(
    n_components=best_params_g["n_components"],
    covariance_type=best_params_g["covariance_type"],
    n_init=best_params_g["n_init"],
    random_state=42
)
gmm.fit(X_PCA)
labels_g = gmm.predict(X_PCA)


# block 4
# KMeans with K-Fold CV
hyperparams_k = {
    "n_clusters": [10],
    "max_iter": [300,400],
    "n_init": [5, 10, 15, 20]
}

kf = KFold(n_splits=3, shuffle=True, random_state=42)
total_k_runs = len(hyperparams_k["n_clusters"]) * len(hyperparams_k["max_iter"]) * len(hyperparams_k["n_init"])
pbar = tqdm(total=total_k_runs, desc="KMeans K-Fold CV")
best_score_k = -1
best_params_k = None

for n_clusters in hyperparams_k["n_clusters"]:
    for max_iter in hyperparams_k["max_iter"]:
        for n_init in hyperparams_k["n_init"]:
            fold_scores = []
            for train_idx, val_idx in kf.split(X_PCA):
                X_train, X_val = X_PCA[train_idx], X_PCA[val_idx]
                model = KMeans(
                    n_clusters=n_clusters,
                    max_iter=max_iter,
                    n_init=n_init,
                    random_state=42
                )
                model.fit(X_train)
                labels_val = model.predict(X_val)
                sil = silhouette_score(X_val, labels_val)
                fold_scores.append(sil)

            avg_sil = np.mean(fold_scores)
            if avg_sil > best_score_k:
                best_score_k = avg_sil
                best_params_k = {"n_clusters": n_clusters, "max_iter": max_iter, "n_init": n_init}
            pbar.update(1)
pbar.close()
print("Best KMeans params:", best_params_k)

kmeans = KMeans(
    n_clusters=best_params_k["n_clusters"],
    max_iter=best_params_k["max_iter"],
    n_init=best_params_k["n_init"],
    random_state=42
)
kmeans.fit(X_PCA)
labels_k = kmeans.labels_


# block 6
# Evaluate clustering performance
print("\nKMEANS METRICS")
print("Adjusted Rand Index KMeans:", adjusted_rand_score(y, labels_k))
print("Adjusted Mutual Info KMeans:", adjusted_mutual_info_score(y, labels_k))
print("Silhouette KMeans:", silhouette_score(X_PCA, labels_k))

print("\nGMM METRICS")
print("Adjusted Rand Index GMM:", adjusted_rand_score(y, labels_g))
print("Adjusted Mutual Info GMM:", adjusted_mutual_info_score(y, labels_g))
print("Silhouette GMM:", silhouette_score(X_PCA, labels_g))


# block 7
# Determine best-performing model for visualization
if best_score_k > best_score_g:
    print("K MEANS")
    labels = labels_k
    K = best_params_k["n_clusters"]
else:
    print("GMM")
    labels = labels_g
    K = best_params_g["n_components"]


# block 8
# Show a grid plot with one row per cluster and five images per cluster
fig, axes = plt.subplots(K, 5, figsize=(15, K*1.5))
for cluster in range(K):
    indices = np.where(labels == cluster)[0]
    num_images = min(5, len(indices))
    chosen = np.random.choice(indices, num_images, replace=False)
    for n, index in enumerate(chosen):
        ax = axes[cluster, n]
        ax.imshow(X[index].reshape(28, 28), cmap="gray")
        ax.axis("off")

if labels == labels_g:
    title = "GMM" 
else: title = "KMeans"
plt.tight_layout()
plt.suptitle(f"Best clustering visualization – {title}", fontsize=16)
plt.show()

