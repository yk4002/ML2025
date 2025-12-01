import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import task4
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier


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

    # training batch
    X_train = np.concatenate(X_all)
    y_train = np.concatenate(y_all)
    # test batch
    X_test, y_test = load_batch(os.path.join(data_dir, "test_batch"))
    return X_train / 255.0, y_train, X_test / 255.0, y_test


# Load the data
data_dir = "cifar-10-batches-py"
X_train, y_train, X_test, y_test = load_cifar10(data_dir)

# Load PCA-reduced inputs
X_train_PCA = task4.X_train_pca
X_test_PCA = task4.X_test_pca

# Define hyperparameters for AdaBoost
# hyperparams = {
#     "n_estimators": [50, 100],
#     "learning_rate": [0.2, 0.4, 0.5],
#     "estimator__max_depth": [3, 5]
# }

#used purely for the purposes of testing - COMMENT OUT IF UNNEEDED
hyperparams = {
    "n_estimators": [200],
    "learning_rate": [0.4],
    "estimator__max_depth": [5]
}



# Base estimator for AdaBoost
base_tree = DecisionTreeClassifier(random_state=42)

# GridSearchCV for AdaBoost
grid = GridSearchCV(
    estimator=AdaBoostClassifier(estimator=base_tree, random_state=42),
    param_grid=hyperparams,
    scoring="accuracy",
    cv=5,
    n_jobs=6,
    verbose=3
)

grid.fit(X_train_PCA, y_train)
opt_hyperparams = grid.best_params_
print("Best hyperparameters:", opt_hyperparams)

# Train AdaBoost with best hyperparameters
model = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=opt_hyperparams["estimator__max_depth"], random_state=42),
    n_estimators=opt_hyperparams["n_estimators"],
    learning_rate=opt_hyperparams["learning_rate"],
    random_state=42
)
model.fit(X_train_PCA, y_train)

# Evaluate on test set
test_accuracy = model.score(X_test_PCA, y_test)
print("Train accuracy:", model.score(X_train_PCA, y_train))
print("Test accuracy:", test_accuracy)

# Visualize 5 misclassified images
f = (15, 6)
y_pred_test = model.predict(X_test_PCA)
misclass_indexes = np.where(y_pred_test != y_test)[0]
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

plt.suptitle("Misclassified images of AdaBoost ensemble classifier")
plt.tight_layout()
plt.show()