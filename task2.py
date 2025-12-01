import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# Load the dataaa
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




#Official Pytorch documentation on building neural nets used as the template this code
# https://docs.pytorch.org/tutorials/beginner/pytorch_with_examples.html

#first all data is converted into tensors for the pytorch
X_train_t = torch.tensor(X_train_p, dtype=torch.float32)
y_train_t = torch.tensor(y_train.to_numpy(), dtype=torch.float32)
X_test_t = torch.tensor(X_test_p, dtype=torch.float32)
y_test_t = torch.tensor(y_test.to_numpy(), dtype=torch.float32)
y_train_t = y_train_t.view(-1, 1)
y_test_t = y_test_t.view(-1, 1)




#hyperparams specified - adjust in case of overfitting/underfitting
lr = 0.001 #learning rate determines number of steps of optimiser
epochs = 10000 #number of iterations

#shape[1] gets the number of columns ie the number of features
#two hidden layers with ReLu function used to get the
# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(X_train_p.shape[1], 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.linear_relu_stack(x)




#the training loop function for the neural net
def training_loop(x, y, model, loss_fn, optimizer, epochs):
    for epoch in range(epochs):
        model.train()
        #forward pass
        ypred = model(x)
        loss = loss_fn(ypred, y)
        print(f"Epoch {epoch}")
        #backwards pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



#Define model
model = NeuralNetwork()
##loss function is MSE
loss_fn = nn.MSELoss()
#Adam is used as the optimiser
adam_opt = torch.optim.Adam(model.parameters(), lr=lr)


#train the neural network
training_loop(X_train_t, y_train_t, model, loss_fn, adam_opt, epochs )


#get predicted y using the model
model.eval()
with torch.no_grad():
    y_train_pred = model(X_train_t)
    y_test_pred = model(X_test_t)
    #convert back to numpy to allow for easy plotting
    y_train_pred.numpy()
    y_test_pred.numpy()


#print RMSE on training and testing set
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
print("RMSE train:", rmse_train)
print("RMSE test:", rmse_test)



# plot predicted versus actual charges on the test set.
plt.figure(figsize=(6,6))
plt.scatter(y_test_pred, y_test)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Predicted charges")
plt.ylabel("Actual charges")            
plt.title("Predicted vs actual for charges")
plt.show()