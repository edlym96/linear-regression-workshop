import numpy as np

def loader(filename):
    c=np.loadtxt('%s.txt' % filename,delimiter=',')
    x=c[:,:-1] #extract every column except the last column
    y=c[:,-1] #extract the last column
    return x,y