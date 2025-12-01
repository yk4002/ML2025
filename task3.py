import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pymc as pm
import arviz as az
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Load the data
file_name = "regression_insurance.csv"
data = pd.read_csv(file_name)

# Split into training and testing sets
X = data.drop(columns=['charges'])
y = data['charges']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# One-hot encoding
categorical_features = ['sex', 'smoker', 'region']

# Use sparse=False for compatibility with all sklearn versions
encoder = OneHotEncoder(drop='first', sparse_output=False)
X_train_c = encoder.fit_transform(X_train[categorical_features])
X_test_c = encoder.transform(X_test[categorical_features])

# Standard scaling
numeric_features = ['age', 'bmi', 'children']
scaler = StandardScaler()
X_train_n = scaler.fit_transform(X_train[numeric_features])
X_test_n = scaler.transform(X_test[numeric_features])

# Concatenate features
X_train_p = np.hstack([X_train_c, X_train_n]).astype("float64")
X_test_p = np.hstack([X_test_c, X_test_n]).astype("float64")
y_train = y_train.astype("float64")

# Build Bayesian regression model
with pm.Model() as model:
    w0 = pm.Normal("w0", mu=0, sigma=20)
    w1 = pm.Normal("w1", mu=0, sigma=20, shape=X_train_p.shape[1])

    # Use HalfNormal instead of Uniform for stability
    sigma = pm.Uniform("sigma", lower=0, upper=10)

    y_est = w0 + pm.math.dot(X_train_p, w1)

    likelihood = pm.Normal("y", mu=y_est, sigma=sigma, observed=y_train.values)

if __name__ == "__main__":
    with model:
        idata = pm.sample(
            draws=200,
            tune=200,
            chains=1,
            cores=-1,
            target_accept=0.9
        )

        # Safe extraction of coefficients
        w1_post = idata.posterior["w1"].mean(dim=("chain", "draw")).values
        sigma_post = idata.posterior["sigma"].mean(dim=("chain", "draw")).values

        print("Noise sigma:", round(float(sigma_post), 3))

        # Build feature names in the correct order
        feature_names_c = encoder.get_feature_names_out(categorical_features)
        feature_names = np.concatenate([feature_names_c, numeric_features])

        for feature, coef in zip(feature_names, w1_post):
            print(f"{feature}: {round(float(coef), 3)}")
