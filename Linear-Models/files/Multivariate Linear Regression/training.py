"""
TRAINING SCRIPT
"""

import numpy as np
from SimpleLinear import simple_linear_model
from MSE import calculate_MSE
from gradient_descent import gradient_descent
from loader import loader
from normalise import normalise
from visualize import visualize

#initialise variables
n_epochs = 50
np.random.seed(0)

#Generate X and y matrices
X,y=loader("ex1data2")
X=normalise(X) #normalise features
m=y.size #get total number of data samples
X=np.hstack((np.ones(m)[:, np.newaxis], X)) #add intercept values to the feature array
n=X.shape[1] #get total number of features
print(X)
print(y)

#Generate Weights
W=np.random.rand(n) #Generate n number of weights
print("weights are:", W)

#Initialise Model
y_hat=simple_linear_model(W,X)

#Commence Training
it=[] #initialise arrays for iterations and losses
loss_array=[]
for _ in range(n_epochs):
    it.append(_)
    print(_)
    #Calculate loss
    loss=calculate_MSE(y,y_hat)
    loss_array.append(loss)
    print("loss is: ",loss)

    #Perform gradient descent
    W = gradient_descent(W,X,y,y_hat,m,0.1)

    #Generate new model
    y_hat = simple_linear_model(W, X)

#plot graph of loss
visualize(it, loss_array)


