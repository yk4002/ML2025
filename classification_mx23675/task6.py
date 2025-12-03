import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import task4
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier


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

# Define hyperparameters for Random Forest
hyperparams = {
    "n_estimators": [100, 200],
    "max_depth": [5,10],
    "max_features": ["sqrt", 0.5],  # fraction or sqrt of the features
    "min_samples_split": [10, 20]
}

# GridSearchCV for Random Forest
grid = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=hyperparams,
    scoring="accuracy",
    cv=3,
    n_jobs=6,
    verbose=3
)

grid.fit(X_train_PCA, y_train)
optparams = grid.best_params_
print(optparams)

# Train Random Forest with best hyperparameters
model = RandomForestClassifier(
    n_estimators=optparams["n_estimators"],
    max_depth=optparams["max_depth"],
    max_features=optparams["max_features"],
    min_samples_split=optparams["min_samples_split"],
    random_state=42
)
model.fit(X_train_PCA, y_train)

# Evaluate on test set
train_accuracy = model.score(X_train_PCA, y_train)
test_accuracy = model.score(X_test_PCA, y_test)
print("Train accuracy:", train_accuracy)
print("Test accuracy:", test_accuracy)

# Visualize 5 misclassified images
f = (15, 6)
y_pred_test = model.predict(X_test_PCA)
misclass_indexes = np.where(y_pred_test != y_test)[0]
np.random.seed(42)
chosen_images = np.random.choice(misclass_indexes, size=5, replace=False)
class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
fig, axes = plt.subplots(1, 5, figsize=f)
for n, index in enumerate(chosen_images):
    ax = axes[n]
    img = X_test[index].reshape(3, 32, 32).transpose(1, 2, 0)
    ax.imshow(img)
    ax.axis("off")
    true_label = class_names[y_test[index]]
    pred_label = class_names[y_pred_test[index]]
    ax.set_title(f"Actual label: {true_label}\nPredicted label: {pred_label}", fontsize=10)

plt.suptitle("Misclassified images of Random Forest classifier")
plt.tight_layout()
plt.show()
