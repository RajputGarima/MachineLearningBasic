import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pylab
import sys

x_file = sys.argv[1]
y_file = sys.argv[2]

x_train = pd.read_csv(x_file,  header=None)
y_train = pd.read_csv(y_file, header=None)
m = x_train.size

mylistX = []
mylistY = []
plotX = []
outputlist = []
outputlistW = []


for i in range(m):
    mylistX.insert(i, [1, x_train.iloc[i][0]])
    plotX.insert(i, x_train.iloc[i][0])
    mylistY.insert(i, y_train.iloc[i][0])


X = np.asarray(mylistX)
print(X)
Y = np.asarray(mylistY)
Z = np.asarray(plotX)


plt.scatter(Z, Y, color='red', marker='x', label='Data points')
plt.xlabel('Input feature')
plt.ylabel('Target variable')
plt.title('Locally Weighted')

# print(X) 2-d array
# print(Y) 1-d array

X_transpose = np.transpose(X)
P = np.dot(X_transpose, X)
P_inverse = np.linalg.pinv(P)
Q = np.dot(P_inverse, X_transpose)
theta = np.dot(Q, mylistY)


def predictY(i):
    temp = theta[0] + theta[1] * X[i][1]
    return temp


for i in range(m):
    outputlist.insert(i, predictY(i))

plotY = np.asarray(outputlist)
plt.plot(Z, plotY, color='blue', label='Linear Regression Hypothesis')
pylab.legend(loc='best')
plt.show()

#---locally weighted---
s = (m, m)
W = np.zeros(s)
tau = float(sys.argv[3])

W1 = np.zeros(s)

max_element = np.amax(Z)
min_element = np.amin(Z)
new_list = []
n = 100
new_array = np.linspace(min_element, max_element, num=100)


def compute_weight_matrix(i):
    x_val = new_array[i]
    for j in range(m):
        t1 = mylistX[j][1] - x_val
        t2 = pow(t1, 2)
        t3 = pow(tau, 2) * 2
        t4 = t2/t3
        W[j][j] = -t4
    global W1
    W1 = np.exp(W)
    for j in range(m):
        for k in range(m):
            if k != j:
                W1[j][k] = 0


def locally_weighted_regression():
    X_transpose = np.transpose(X)
    P = np.dot(X_transpose, W1)
    Q = np.dot(P, X)
    Q_inverse = np.linalg.pinv(Q)
    R = np.dot(Q_inverse, X_transpose)
    S = np.dot(R, W1)
    global theta
    theta = np.dot(S, Y)


def predictWY(i):
    temp = theta[0] + theta[1] * new_array[i]
    return temp


for i in range(n):
    compute_weight_matrix(i)
    locally_weighted_regression()
    outputlistW.insert(i, predictWY(i))

plotY = np.asarray(outputlistW)
plotX = np.asarray(new_array)
plt.scatter(Z, Y, color='red', marker='x', label='Data points')
plt.xlabel('Input feature')
plt.ylabel('Target variable')
plt.title('Locally Weighted')
plt.plot(plotX, plotY, color='green', label='Locally Weighted Hypothesis')
pylab.legend(loc='best')
plt.show()
