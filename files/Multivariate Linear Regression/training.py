"""
TRAINING SCRIPT
"""

import numpy as np
from SimpleLinear import simple_linear_model
from MSE import calculate_MSE
from gradient_descent import gradient_descent
from gradient_descent import regularised_gradient_descent
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

#Split the data set into training set, cross-validation set and test set
X=np.vsplit(X,[int(0.6*X.shape[0]),int(0.8*X.shape[0])])
y=np.split(y,[int(0.6*y.size),int(0.8*y.size)])

X_train=X[0]
X_cv=X[1]
X_test=X[2]

y_train=y[0]
y_cv=y[1]
y_test=y[2]

#Generate Weights
W=np.random.rand(n) #Generate n number of weights
print("weights are:", W)

#Initialise Model
y_hat=simple_linear_model(W,X_train)

#Commence Training
it=[] #initialise arrays for iterations and losses
loss_array=[]
for _ in range(n_epochs):
    it.append(_)
    print(_)
    #Calculate loss
    loss=calculate_MSE(y_train,y_hat)
    loss_array.append(loss)
    print("loss is: ",loss)

    #Perform gradient descent
    W = gradient_descent(W,X_train,y_train,y_hat,m,0.1)

    #Generate new model
    y_hat = simple_linear_model(W, X_train)

#plot graph of loss
visualize(it, loss_array)

y_hat_cv=simple_linear_model(W,X_cv)
loss_cv=calculate_MSE(y_cv,y_hat_cv)

print("training loss: ", loss)
print("cross validation loss: ", loss_cv)

pause = input('PRESS ENTER TO CONTINUE')

W_reg=np.random.rand(n)
y_hat_reg=simple_linear_model(W_reg,X_train)
for _ in range(n_epochs):
    it.append(_)
    print(_)
    #Calculate loss
    loss_reg=calculate_MSE(y_train,y_hat_reg)
    print("loss is: ",loss_reg)

    #Perform gradient descent
    W_reg = regularised_gradient_descent(W,X_train,y_train,y_hat_reg,m,0.1,0.01)

    #Generate new model
    y_hat_reg = simple_linear_model(W_reg, X_train)

y_hat_cv_reg=simple_linear_model(W_reg,X_cv)
loss_cv_reg=calculate_MSE(y_cv,y_hat_cv_reg)

print("regularised training loss: ", loss_reg)
print("regularised cross validation loss: ", loss_cv_reg)