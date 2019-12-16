import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import pylab
import sys

x_file = sys.argv[1]
y_file = sys.argv[2]

modem = sys.argv[3]


data_x = np.loadtxt(x_file)
f = open(y_file, mode='r')
output = []
index = 0

for line in f:
    line = line.rstrip()
    if line == "Alaska":    # Alaska --> 0 , Canada --> 1
        output.insert(index, 0)
    else:
        output.insert(index, 1)
    index = index + 1
f.close()

data_y = np.asarray(output)
m = data_y.size
mu_vector_size = (2, 1)
mu_0 = np.zeros(mu_vector_size)
mu_1 = np.zeros(mu_vector_size)
rs = (2, 2)
z = (2, 1)
covariance = np.zeros(rs)
phi = 0
normalised_x = []
class0X = []
class0Y = []
class1X = []
class1Y = []
index0 = 0
index1 = 0
theta_zero = 0.0
theta_one = 0.0
theta_two = 0.0


def normalise():
    mean = np.mean(data_x, axis=0)
    st_dev = np.std(data_x, axis=0)
    for i in range(m):
        normalised_x.insert(i, [(data_x[i][0] - mean[0])/st_dev[0], (data_x[i][1] - mean[1])/st_dev[1]])


normalise()
data_xn = np.asarray(normalised_x)


def classify_plot():
    global index0, index1
    for i in range(m):
        if data_y[i] == 0:
            class0X.insert(index0, data_xn[i][0])
            class0Y.insert(index0, data_xn[i][1])
            index0 = index0 + 1
        else:
            class1X.insert(index1, data_xn[i][0])
            class1Y.insert(index1, data_xn[i][1])
            index1 = index1 + 1


classify_plot()


def mu_0_func():
    count = 0
    sum1 = 0.0
    sum2 = 0.0
    for i in range(m):
        if data_y[i] == 0:
            sum1 = sum1 + data_xn[i][0]
            sum2 = sum2 + data_xn[i][1]
            count = count + 1
    mu_0[0][0] = sum1/count
    mu_0[1][0] = sum2/count


def mu_1_func():
    sum1 = 0
    sum2 = 0
    count = 0
    for i in range(m):
        if data_y[i] == 1:
            sum1 = sum1 + data_xn[i][0]
            sum2 = sum2 + data_xn[i][1]
            count = count + 1
    mu_1[0][0] = sum1/count
    mu_1[1][0] = sum2/count


def sigma():
    global covariance, z
    inter = np.zeros(z)
    for i in range(m):
        if data_y[i] == 1:
            inter[0][0] = data_xn[i][0]
            inter[1][0] = data_xn[i][1]
            temp = inter - mu_1
            covariance = covariance + np.dot(temp, np.transpose(temp))
        elif data_y[i] == 0:
            inter[0][0] = data_xn[i][0]
            inter[1][0] = data_xn[i][1]
            temp = inter - mu_0
            covariance = covariance + np.dot(temp, np.transpose(temp))
    covariance = covariance/m


def phi_func():
    count = 0
    global phi
    for i in range(m):
        if data_y[i] == 1:
            count = count + 1
    phi = count/m


def gauss_distribution(i, mu):
    inter = np.zeros(z)
    inter[0][0] = data_xn[i][0]
    inter[1][0] = data_xn[i][1]
    temp1 = inter - mu
    temp2 = np.dot(np.transpose(temp1), covariance)
    temp3 = np.dot(temp2, temp1)
    temp4 = -0.5 * temp3
    temp5 = np.exp(temp4)
    determinant_power_half = pow(np.linalg.det(covariance), 0.5)
    temp6 = 2 * math.pi * determinant_power_half
    return temp5[0][0]/temp6


def prob_x_given_y_equal_1(i):
    return gauss_distribution(i, mu_1)


def prob_x_given_y_equal_0(i):
    return gauss_distribution(i, mu_0)


def prob_x(i):
    return prob_x_given_y_equal_1(i) * phi + prob_x_given_y_equal_0(i) * (1 - phi)


def prob_y_equal_one_given_x(i):
    return (prob_x_given_y_equal_1(i) * phi)/prob_x(i)


def prob_y_equal_zero_given_x(i):
    return (prob_x_given_y_equal_0(i) * (1 - phi))/prob_x(i)


plt.scatter(class0X, class0Y, color='red', s=10**1.50, marker='x', label='Alaska')
plt.scatter(class1X, class1Y, color='blue', label='Canada')
plt.xlabel('Input feature(X1)')
plt.ylabel('Input feature(X2)')
plt.title('Gaussian Discriminant Analysis')
if modem == '0':
    pylab.legend(loc='best')
    plt.show()

mu_0_func()
mu_1_func()
sigma()
phi_func()
if modem == '0':
    print('Parameters of part(a):')
    print(mu_0)
    print(mu_1)
    print(covariance)


def compute_line_theta():
    global theta_one, theta_two, theta_zero
    temp = np.linalg.pinv(covariance)
    t1 = np.dot(np.transpose(mu_0),temp)
    t2 = np.dot(t1, mu_0)
    t3 = np.dot(np.transpose(mu_1), temp)
    t4 = np.dot(t3, mu_1)
    t6 = (t2 - t4) * 0.5
    t7 = math.log(((1/phi)-1), math.e)
    theta_zero = t6[0][0] - t7
    t5 = np.dot(temp, (mu_1 - mu_0))
    theta_one = t5[0][0]
    theta_two = t5[1][0]


plot_listX = []
plot_listY = []


def plot_line():
    compute_line_theta()
    for i in range(m):
        plot_listY.insert(i, -(theta_zero/theta_two) - (theta_one/theta_two) * data_xn[i][0])
        plot_listX.insert(i, data_xn[i][0])


plot_line()
plt.plot(plot_listX, plot_listY, color='green', linewidth='1.5', label='Linear decision boundary')
if modem == '0':
    plt.scatter(class0X, class0Y, color='red', s=10 ** 1.50, marker='x', label='Alaska')
    plt.scatter(class1X, class1Y, color='blue', label='Canada')
    plt.xlabel('Input feature(X1)')
    plt.ylabel('Input feature(X2)')
    plt.title('Gaussian Discriminant Analysis')
    pylab.legend(loc='best')
    plt.show()

sigma0 = np.zeros(rs)
sigma1 = np.zeros(rs)


def quad_covariance():
    c0 = 0
    c1 = 0
    global sigma0, sigma1
    inter = np.zeros(z)
    for i in range(m):
        if data_y[i] == 0:
            c0 = c0 + 1
            inter[0][0] = data_xn[i][0]
            inter[1][0] = data_xn[i][1]
            temp = inter - mu_0
            sigma0 = sigma0 + np.dot(temp, np.transpose(temp))
        else:
            c1 = c1 + 1
            inter[0][0] = data_xn[i][0]
            inter[1][0] = data_xn[i][1]
            temp = inter - mu_1
            sigma1 = sigma1 + np.dot(temp, np.transpose(temp))
    sigma0 = sigma0/c0
    sigma1 = sigma1/c1


quad_covariance()
if modem == '1':
    print('Parameters of part(d):')
    print(mu_0)
    print(mu_1)
    print(sigma0)
    print(sigma1)

x = np.zeros(rs)
y = np.zeros(z)
z = 0.0


def quad_boundary_parameters():
    global x, y, z
    inv1 = np.linalg.pinv(sigma0)
    inv2 = np.linalg.pinv(sigma1)
    x = inv1 - inv2
    mu0_inv1 = np.dot(np.transpose(mu_0), inv1)
    mu_1_inv2 = np.dot(np.transpose(mu_1), inv2)
    y = -2 * (mu0_inv1 - mu_1_inv2)
    t1 = np.dot(mu0_inv1, mu_0) - np.dot(mu_1_inv2, mu_1)
    t2 = 2 * np.log(((1/phi)-1) * (np.linalg.det(sigma1)/np.linalg.det(sigma0)))
    z = t1 - t2


quad_boundary_parameters()
plot_list_x2_1 = []
plot_list_x2_2 = []
plot_listX = []
index = 0


def predicty(x1):
    global index
    L = pow(x1, 2) * x[0][0] + y[0][0] * x1 + z[0][0]
    M = x1 * x[1][0]
    N = x1 * x[0][1]
    O = M + N
    P = O + y[0][1]
    dis = pow(P, 2) - 4 * x[1][1] * L
    x2_a = (-P + math.sqrt(dis))/(2 * x[1][1])
    x2_b = (-P - math.sqrt(dis))/(2 * x[1][1])
    plot_list_x2_1.insert(index, x2_a)
    plot_list_x2_2.insert(index, x2_b)
    plot_listX.insert(index, x1)
    index = index + 1


def quad_boundary():
    for i in range(m):
        predicty(data_xn[i][0])


p, q = np.mgrid[-2.5:2.5:50j, -3:3:50j]
M = np.c_[p.flatten(), q.flatten()]

A = x
B = y
C = z


def bdry(x):
    return x.T @ A @ x + B @ x + C


if modem == '1':
    quad = np.array([bdry(m) for m in M]).reshape(p.shape)
    plt.contour(p, q, quad, [0], colors='black')
    quad_boundary()
    pylab.legend(loc='best')
    plt.show()

#plt.plot(plot_listX, plot_list_x2_1, color='yellow', linewidth='0.50', label='Quadratic Decision Boundary')
#plt.plot(plot_listX, plot_list_x2_2, color='yellow', linewidth='0.50')
