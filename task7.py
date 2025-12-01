import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import task4


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

X_train_PCA = task4.X_train_pca
X_test_PCA = task4.X_test_pca


# Use cross-
# validation to choose hyperparameters such as kernel type, C, and γ. 
#have a look at what other hyperparams might be needed for SVM through equation? And their range of stuff
hyperparams = {
    "kernel": ['linear', 'rbf'],
    "C": [0.1, 1, 10],
    "gamma": ['scale', 0.01, 0.1]
}


#PURELY FOR TESTING PURPOSES
hyperparams = {
    "kernel": ['linear'],
    "C": [0.1, 1, 10],
    "gamma": ['scale', 0.01, 0.1]
}



grid = GridSearchCV(
    estimator= SVC(),  
    param_grid=hyperparams, 
    scoring="accuracy",
    cv = 5, #might need to talk about this
    n_jobs=-1,
    verbose=3,  
)
grid.fit(X_train_PCA, y_train)
optparams = grid.best_params_


# Train a Support Vector Machine (SVM) classifier. 
model = SVC(kernel = optparams["kernel"], C=optparams["C"], gamma = optparams["gamma"], )
model.fit(X_train_PCA, y_train)


# Evaluate the final SVM model on the test set and report the test accuracy
test_accuracy = model.score(X_test_PCA, y_test)
print("Train accuracy:", model.score(X_train_PCA, y_train))
print("Test accuracy:", test_accuracy)

#visualise 5 misclassified images? 
#check the examples find this code somwhere else and link the source please
f = (15,6)
y_pred_test = model.predict(X_test_PCA)
misclass_indexes = np.where(y_pred_test != y_test)[0] #ie misclassified
np.random.seed(42)
chosen_images = np.random.choice(misclass_indexes, size=5, replace=False)
class_names = [
    'airplane','automobile','bird','cat','deer',
    'dog','frog','horse','ship','truck'
]

fig, axes = plt.subplots(1, 5, figsize=f)
for n, index in enumerate(chosen_images):
    ax = axes[n]
    img = X_test[index].reshape(3, 32, 32).transpose(1, 2, 0)
    ax.imshow(img)
    ax.axis("off")
    true_label = class_names[y_test[index]]
    pred_label = class_names[y_pred_test[index]]
    ax.set_title(f"T: {true_label}\nP: {pred_label}", fontsize=10)

plt.suptitle("Misclassified images of SVM classifier")
plt.tight_layout()
plt.show()