import pickle
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def load_batch(batch_path):
    with open(batch_path, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
        X = batch[b'data']
        y = np.array(batch[b'labels'])
        return X, y


def load_cifar10(data_dir):
    X_all = []
    y_all = []
    # 5 training batches
    for i in range(1, 6):
        batch_path = os.path.join(data_dir, f"data_batch_{i}")
        X, y = load_batch(batch_path)
        X_all.append(X)
        y_all.append(y)

    #training batch
    X_train = np.concatenate(X_all)
    y_train = np.concatenate(y_all)
    # test batch
    X_test, y_test = load_batch(os.path.join(data_dir, "test_batch"))
    return X_train/255, y_train, X_test/255, y_test


# Load the data
# You may need to specify ‘data_dir’ the directory of dataset folder
data_dir = "cifar-10-batches-py"
X_train, y_train, X_test, y_test = load_cifar10(data_dir)


# sample is 3072 dimensional, reduce dimension to 200 features using PCA
#Source??

n = 200 #the number of features we wanna reduce to
#centre the training and testing data without scaling 
scaler = StandardScaler(with_std=False) #is that argument necessary?
X_train_centered = scaler.fit_transform(X_train)
X_test_centered = scaler.transform(X_test)

#do pca and reduce to n components
pca = PCA(n_components=n)
X_train_pca = pca.fit_transform(X_train_centered)
X_test_pca = pca.transform(X_test_centered)

#PCA requires zero-mean data.
# Scaling pixels to unit variance is not appropriate for images, because it artificially amplifies noise in low-variance channels.
# Thus, centered-only PCA is the correct design choice.