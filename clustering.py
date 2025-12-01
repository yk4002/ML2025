#USE THIS AS THE THING TO RUN OVERNIGHT

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_mutual_info_score, adjusted_rand_score
from tqdm import tqdm

#block 1
# IMPORTANT Load the MNIST dataset
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist["data"], mnist["target"].astype(int)


#block 2
# Visualize sampled images and labels.
# grid plot: one row per class and five images per class.
# (Source inspiration: official sklearn MNIST examples + typical MNIST visualization snippets) LINK??
f=(15,10)
classes = 10 #classes from 0-9
imgs_per_class = 5
fig, axes = plt.subplots(classes, imgs_per_class, figsize=f)
#iterate through every class
for mnist_dig in range(classes):
    indexes = np.where(y==mnist_dig)[0] #check when the y values equal the digit
    chosen = np.random.choice(indexes, imgs_per_class, replace=False) #get a bunch of images per class
    #now iterate through the chosen ones
    for i, idx in enumerate(chosen):
        ax = axes[mnist_dig, i] #create two axes?
        ax.imshow(X[idx].reshape(28, 28), cmap="gray") #reshape to size of the MNIST
        ax.axis("off") #readability

plt.suptitle("Sampled MNIST images with class labels", fontsize=20)
plt.show()



#block 3
#dimensionality reduction to 100 features so that training is faster
n = 100
scaler = StandardScaler(with_std=False)
X_centred = scaler.fit_transform(X)
pca = PCA(n_components=n)
X_PCA = pca.fit_transform(X_centred)


#block 4
#SOURCE: Scikit Silhouette Analysis for KMeans Clustering (and GMM) where it is used for one parameter - here we use it for 3!
#use silhouette score to evaluate Kmeans. Maybe explore one more hyperparameter?
hyperparams_k = {
    "n_clusters": [10],
    "max_iter": [300,400],
    "n_init": [5, 10, 15, 20] #n_init specifies how many times the K-Means algorithm will run with different centroid seeds.
}

#progress bar setup 
total_k_runs = (
    len(hyperparams_k["n_clusters"]) *
    len(hyperparams_k["max_iter"]) *
    len(hyperparams_k["n_init"])
)
pbar = tqdm(total=total_k_runs, desc="KMeans search")


#loop through all hyperparam combinatiosn and use silhouette score as the matrix to evaluate which one is the best
best_score_k = -1
best_params_k = None
for n_clusters in hyperparams_k["n_clusters"]:
    for max_iter in hyperparams_k["max_iter"]:
        for n_init in hyperparams_k["n_init"]:
            model = KMeans(n_clusters=n_clusters, max_iter=max_iter, n_init=n_init, random_state=42)
            labels = model.fit_predict(X_PCA)
            sil_sc = silhouette_score(X_PCA, labels)
            if sil_sc > best_score_k:
                best_score_k = sil_sc
                best_params_k = {"n_clusters": n_clusters, "max_iter": max_iter, "n_init": n_init}
            pbar.update(1)
pbar.close()
print("Best KMeans params:", best_params_k)

#define the model with optimised paraneters
kmeans = KMeans(n_clusters=best_params_k["n_clusters"], 
                max_iter=best_params_k["max_iter"],
                n_init=best_params_k["n_init"],
                random_state=42)
kmeans.fit(X_PCA)
labels_k = kmeans.labels_



#block 5
#GMM
hyperparams_gmm = {
    "n_components": [10], #number of clusters - ideally 10 but who knows
    "covariance_type": ["full", "diag"], ##changes how clusters fitted into shapes. Spherical too simple for MNIST?
    "n_init": [5,10] #how many times run with different configs
}

total_g_runs = (
    len(hyperparams_gmm["n_components"]) *
    len(hyperparams_gmm["covariance_type"]) *
    len(hyperparams_gmm["n_init"])
)

pbar = tqdm(total=total_g_runs, desc="GMM search")


best_score_g = -1
best_params_g = None

#iterate through the loop
for n_comp in hyperparams_gmm["n_components"]:
    for cov in hyperparams_gmm["covariance_type"]:
        for n_init in hyperparams_gmm["n_init"]:
            gmm = GaussianMixture(n_components=n_comp,covariance_type=cov,n_init=n_init,random_state=42)
            gmm.fit(X_PCA)
            labels = gmm.predict(X_PCA)
            #use the silhouette score to evaluate the stuff
            sil = silhouette_score(X_PCA, labels)
            if sil > best_score_g:
                best_score_g = sil
                best_params_g = {"n_components": n_comp,"covariance_type": cov,"n_init": n_init}
            pbar.update(1)
pbar.close()
            
print("Best GMM params:", best_params_g)

#initialise models with ideal params, fit to data and then get labels
gmm = GaussianMixture(n_components=best_params_g["n_components"],
                      covariance_type=best_params_g["covariance_type"],
                      n_init=best_params_g["n_init"],
                      random_state=42)
gmm.fit(X_PCA)
labels_g = gmm.predict(X_PCA)



#block 6
# Evaluate clustering performance of both clustering methods
print("\nKMEANS METRICS")
print("Adjusted Rand Index Kmeans:", adjusted_rand_score(y, labels_k))
print("Adjusted Mutual Info Kmeans:", adjusted_mutual_info_score(y, labels_k))
print("Silhouette Kmeans:", silhouette_score(X_PCA, labels_k))

print("\nGMM METRICS")
print("Adjusted Rand Index GMM:", adjusted_rand_score(y, labels_g))
print("Adjusted Mutual Info GMM:", adjusted_mutual_info_score(y, labels_g))
print("Silhouette GMM:", silhouette_score(X_PCA, labels_g))



#block 7
# IMPORTANT: Visualize clustering results. Using the best-performing model and configuration. 
# Show a grid plot with one row per cluster and five images per cluster.
if best_score_k > best_score_g:
    print("K MEANS")
    labels = labels_k
    title = "KMeans"
    K = best_params_k["n_clusters"]
else:
    print("GMM")
    labels = labels_g
    title = "GMM"
    K = best_params_g["n_components"]


#NOT SURE I UNDErSTAND THIS
#graphing the subplots
fig, axes = plt.subplots(K, 5, figsize=(15, K*1.5))
for cluster in range(K):
    #first find the indices where the labels equal the cluster in K?
    indices = np.where(labels == cluster)[0]
    #then do random choicem accounting for when the  for when cluster has less than 5
    num_images = min(5, len(indices))
    chosen = np.random.choice(indices, num_images, replace=False)

    
    #reshape - 
    for n, index in enumerate(chosen):
        ax = axes[cluster, n]
        ax.imshow(X[index].reshape(28, 28), cmap="gray") #reshape it back to original dimensions
        ax.axis("off")
        ax.set_title(f"Img {index}", fontsize=8)

plt.tight_layout()
plt.title(f"Best clustering visualization – {title}")
plt.show()
