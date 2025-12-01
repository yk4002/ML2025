import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the data
file_name = "regression_insurance.csv"
data = pd.read_csv(file_name)
# Split into training and testing sets (80% / 20%)
X = data.drop(columns=['charges'])
y = data['charges']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



#use one hot encoding, standard scaler on the categorical and numerical features
categorical_features = ['sex', 'smoker', 'region']
encoder = OneHotEncoder(drop='first', sparse_output = False)
X_train_c = encoder.fit_transform(X_train[categorical_features])
X_test_c = encoder.transform(X_test[categorical_features])
numeric_features = ['age', 'bmi', 'children']
scaler = StandardScaler()
X_train_n = scaler.fit_transform(X_train[numeric_features])
X_test_n = scaler.transform(X_test[numeric_features])


#Concatenate features before training the model.
X_train_p = np.hstack([X_train_c, X_train_n])
X_test_p = np.hstack([X_test_c, X_test_n])


# Train a linear regression model to predict the insurance charges.
model = LinearRegression()
model.fit(X_train_p, y_train)


# Print the learned coefficients for each feature
coefficients = model.coef_
feature_names_c = encoder.get_feature_names_out(categorical_features)
feature_names = np.concatenate([feature_names_c, numeric_features])
f_c = pd.Series(data=model.coef_, index=feature_names)
for feature, coef in f_c.items():
    print(f"{feature}: {round(coef, 3)}")



# Compute and print RMSE on both training and test sets. 
y_train_pred = model.predict(X_train_p)
y_test_pred = model.predict(X_test_p)
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
print("RMSE for training data:", rmse_train)
print("RMSE for testing data:", rmse_test )


# Produce a scatter plot of predicted versus actual charges on the test set
plt.figure(figsize=(6,6))
plt.scatter(y_test_pred, y_test)
min_val = min(y_test.min(), y_test_pred.min())
max_val = max(y_test.max(), y_test_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--')
plt.xlabel("Predicted charges")
plt.ylabel("Actual charges")            
plt.title("Predicted vs actual charges (using Test Set)")
plt.show()

