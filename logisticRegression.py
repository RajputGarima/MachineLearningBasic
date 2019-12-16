import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import pylab
import sys

x_file = sys.argv[1]
y_file = sys.argv[2]

x_train = pd.read_csv(x_file, header=None)
y_train = pd.read_csv(y_file, header=None)
m = y_train.size
# print(x_train.iloc[0])

class0X = []
class0Y = []
class1X = []
class1Y = []
outputlist = []
mylistX = []
theta_size = (3, 1)
hessian_size = (3, 3)
f_dash_size = (3, 1)
epsilon = 0.0001
index0 = 0
index1 = 0

for i in range(m):
    mylistX.insert(i, x_train.iloc[i][0])

for i in range(m):
    if y_train.iloc[i][0] == 0:
        class0X.insert(index0, x_train.iloc[i][0])
        class0Y.insert(index0, x_train.iloc[i][1])
        index0 = index0 + 1
    else:
        class1X.insert(index1, x_train.iloc[i][0])
        class1Y.insert(index1, x_train.iloc[i][1])
        index1 = index1 + 1

plt.scatter(class0X, class0Y, color='orange', marker='^', label='Class 0')
plt.scatter(class1X, class1Y, color='black', label='Class 1')
plt.xlabel('Input feature (X1)')
plt.ylabel('Input feature (X2)')
plt.title('Logistic Regression')

theta_t = np.zeros(theta_size)
theta_t_plus_one = np.zeros(theta_size)
hessian = np.zeros(hessian_size)
f_dash_theta = np.zeros(f_dash_size)


def h_theta(inp):
    x = np.zeros(theta_size)
    x[0] = 1
    x[1] = inp[0]
    x[2] = inp[1]
    theta_t_transpose = np.transpose(theta_t)
    theta_transpose_x = np.dot(theta_t_transpose, x)
    # theta_transpose_x[0][0] = - theta_transpose_x[0][0]
    # temp1 = np.exp(theta_transpose_x)
    if theta_transpose_x[0][0] >= 0:
        temp1 = math.exp(-theta_transpose_x[0][0])
        return 1/(1 + temp1)
    else:
        temp1 = math.exp(theta_transpose_x)
        return temp1/(1 + temp1)


def del_theta(j):
    result = 0.0
    if j > 0:
        for i in range(m):
            result = result + (y_train.iloc[i][0] - h_theta(x_train.iloc[i])) * x_train.iloc[i][j-1]
    else:
        for i in range(m):
            result = result + y_train.iloc[i][0] - h_theta(x_train.iloc[i])
    return result


def del_theta_square(j, k):
    result = 0.0
    if j == 0 and k == 0:
        for i in range(m):
            temp = h_theta(x_train.iloc[i])
            result = result + temp * (1 - temp)
    elif j == 0:
        for i in range(m):
            temp = h_theta(x_train.iloc[i])
            result = result + (temp * (1 - temp) * x_train.iloc[i][k-1])
    elif k == 0:
        for i in range(m):
            temp = h_theta(x_train.iloc[i])
            result = result + (temp * (1 - temp) * x_train.iloc[i][j-1])
    else:
        for i in range(m):
            temp = h_theta(x_train.iloc[i])
            result = result + (temp * (1 - temp) * x_train.iloc[i][j-1] * x_train.iloc[i][k-1])
    return result


def convergence(flag):
    if flag:
        return True
    else:
        temp = theta_t_plus_one - theta_t
        if temp[0][0] < epsilon and temp[1][0] < epsilon and temp[2][0] < epsilon:
            return False
        else:
            return True


def newton_method():
    flag = True
    while convergence(flag):
        global theta_t_plus_one, theta_t
        theta_t = theta_t_plus_one
        for i in range(3):
            f_dash_theta[i][0] = del_theta(i)
        for i in range(3):
            for j in range(3):
                hessian[i][j] = del_theta_square(i, j)
        hessian_inverse = np.linalg.pinv(hessian)
        temp1 = np.dot(hessian_inverse, f_dash_theta)
        theta_t_plus_one = theta_t - temp1
        flag = False


newton_method()


def predict_y(i):
    temp = -(theta_t_plus_one[0]/theta_t_plus_one[2]) - (theta_t_plus_one[1]/theta_t_plus_one[2])*x_train.iloc[i][0]
    return temp


print(theta_t_plus_one)

for i in range(m):
    outputlist.insert(i, predict_y(i))

plt.plot(mylistX, outputlist, color='green', label='Decision Boundary')
pylab.legend(loc='best')
plt.show()
