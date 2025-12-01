import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
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
data_dir = "cifar-10-batches-py"
X_train, y_train, X_test, y_test = load_cifar10(data_dir)


#import the dimensionality reduced inputs of this cifar set and use them
#PCA already done to reduce the x to 200 features in task 4
X_train_PCA = task4.X_train_pca
X_test_PCA = task4.X_test_pca



#SOURCE: scikit documentation for tuning hyperparams of estimater
#https://scikit-learn.org/stable/modules/grid_search.html#


#define the values of each hyperparam to be explored
hyperparams = {
    "max_depth": [5, 10, 15],
    "min_samples_split": [10, 20, 50],
    "min_samples_leaf": [5, 10, 20]
}


#define the grid
grid = GridSearchCV(
    estimator= DecisionTreeClassifier(random_state = 42),  
    param_grid=hyperparams, 
    scoring="accuracy",
    cv = 2, #talk about this   
    n_jobs=6,   
    verbose=3,                      
)
#find the hyperparams which fit the data the best
grid.fit(X_train_PCA, y_train)
opt_hyperparams = grid.best_params_
print(opt_hyperparams)


# Using the data after dimension reduction, train a single decision
# tree classifier on the CIFAR–10 training set. 
model = DecisionTreeClassifier(min_samples_leaf = opt_hyperparams["min_samples_leaf"], 
                               max_depth=opt_hyperparams["max_depth"], 
                               min_samples_split=opt_hyperparams["min_samples_split"],
                               random_state=42)
model.fit(X_train_PCA, y_train)


# Evaluate the final model on the official test set and report the test
# accuracy
test_accuracy = model.score(X_test_PCA, y_test)
print("Train accuracy:", model.score(X_train_PCA, y_train))
print("Test accuracy:", test_accuracy)


#visualise 5 misclassified images? 
#SOURCE: https://www.cs.toronto.edu/~kriz/cifar.html.
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

plt.suptitle("Misclassified images of Decision Tree classifier")
plt.tight_layout()
plt.show()




