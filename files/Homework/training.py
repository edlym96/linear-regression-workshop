import numpy as np
from multi_linear import multi_linear_model
from MSE import calculate_MSE
from gradient_descent import gradient_descent
from loader import loader
from normalise import normalise
from visualize import visualize

#initialise variables
n_epochs = 500 #epochs means the number of iterations the training runs through the data set
np.random.seed(0)

#Generate X and y matrices
X,y=loader("fertyld")
X=normalise(X)

m=y.size #get total number of data samples
X=np.hstack((np.ones(m)[:, np.newaxis], X)) #add intercept values to the feature array
n=X.shape[1] #get total number of features
print(X)

#Generate Random N Weights
W=np.random.rand(n)

#Initialise Model
y_hat=multi_linear_model(W,X)

for _ in range(n_epochs):
    print(_)
    #Calculate loss
    loss=calculate_MSE(y,y_hat)
    print("loss is: ",loss)

    #Perform gradient descent
    W = gradient_descent(W,X,y,y_hat,m,0.1)

    #Generate new model
    y_hat = multi_linear_model(W, X)

visualize(X[:,1],y,y_hat)

#TRY POLYNOMIAL
power=3
for i in range(1,power):
    new_col=np.reshape(np.power(X[:,1],i+1),(X.shape[0],1))
    X=np.append(X,normalise(new_col),axis=1)

n=X.shape[1] #get total number of features
#Generate Random N Weights
W=np.random.rand(n)

#Initialise Model
y_hat=multi_linear_model(W,X)

for _ in range(n_epochs):
    print(_)
    #Calculate loss
    loss=calculate_MSE(y,y_hat)
    print("loss is: ",loss)

    #Perform gradient descent
    W = gradient_descent(W,X,y,y_hat,m,0.1)

    #Generate new model
    y_hat = multi_linear_model(W, X)
print(W)
visualize(X[:,1],y,y_hat)