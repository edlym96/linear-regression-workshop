"""
TRAINING SCRIPT
"""

import numpy as np
from SimpleLinear import simple_linear_model
from MSE import calculate_MSE
from gradient_descent import gradient_descent
from visualize import visualize
from loader import loader

n_epochs = 15000 #number of iterations
np.random.seed(0) #seeding to persist results

#Load x and y values
x,y=loader('ex1data1')
x=x.flatten()
print("x is: ",x)
print("y is: ",y)

m=y.size #Get total number of data samples

#Generate weights
w0=np.random.rand(1)
w1=np.random.rand(1)
print("weights are: ", w0,w1)

#Initialise Model
y_hat=simple_linear_model(w0,w1,x)
print("y_hat is: ",y_hat)

#Commence Training
for _ in range(n_epochs):
    print(_) #print iteration number

    #Calculate loss
    loss=calculate_MSE(y,y_hat)
    print("loss is: ",loss)

    #Perform gradient descent to update weights
    w0,w1 = gradient_descent(w0,w1,x,y,y_hat,m,0.01)

    #Generate new model
    y_hat = simple_linear_model(w0, w1, x)

#Plot the model
visualize(x, y, y_hat)


