import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm, pylab
import sys

x_file = sys.argv[1]
y_file = sys.argv[2]

x_train = pd.read_csv(x_file,  header=None)
y_train = pd.read_csv(y_file, header=None)
x_size = x_train.size

s = (2, 1)
theta_vector = np.zeros(s)
alpha = float(sys.argv[3])
time_gap = float(sys.argv[4])
prev = 0.0
epsilon = 10 ** -15
input_x = []
input_y = []
output_y = []
normalised_x = []
theta0_mesh = []
theta1_mesh = []
j_theta_mesh = []
index = 0
converged = True

for i in range(x_size):
    input_x.insert(i, x_train.iloc[i][0])
    input_y.insert(i, y_train.iloc[i][0])

input_x_np = np.asarray(input_x)


def normalise():
    global input_x_np
    mean_x = np.mean(input_x_np, axis=0)
    st_dev_x = np.std(input_x_np, axis=0)
    for i in range(x_size):
        normalised_x.insert(i, (input_x_np[i] - mean_x)/st_dev_x)


normalise()
input_x = np.asarray(normalised_x)

plt.scatter(input_x, input_y, color='red', marker='x', label='Data points')


def h_theta(i):
    theta_transpose = np.transpose(theta_vector)
    x = np.zeros(s)
    x[0][0] = 1
    x[1][0] = input_x[i]
    p = np.dot(theta_transpose, x)
    return p[0][0]


def del_theta_zero():
    error = 0.0
    for i in range(x_size):
        error = error - (input_y[i] - h_theta(i))
    return error/x_size


def del_theta_one():
    error = 0.0
    for i in range(x_size):
        error = error - (input_y[i] - h_theta(i)) * input_x[i]
    return error/x_size


def j_theta():
    error = 0.0
    for i in range(x_size):
        error = error + pow((input_y[i] - h_theta(i)), 2)
    return error/(2*x_size)


def convergence():
    global prev, index
    curr = j_theta()
    theta0_mesh.insert(index, theta_vector[0][0])
    theta1_mesh.insert(index, theta_vector[1][0])
    j_theta_mesh.insert(index, curr)
    index = index + 1
    error = abs(curr - prev)
    prev = curr
    if error < epsilon:
        return False
    else:
        return True


def linear_regression():
    count = 10 ** 4
    t = 0
    while convergence() and t < count:
        t = t + 1
        temp0 = theta_vector[0][0] - alpha * del_theta_zero()
        temp1 = theta_vector[1][0] - alpha * del_theta_one()
        theta_vector[0][0] = temp0
        theta_vector[1][0] = temp1
    if t >= count:
        global converged
        converged = False


linear_regression()
print(theta_vector)

for i in range(x_size):
    output_y.insert(i, h_theta(i))

plt.plot(input_x, output_y, color='green', label='Hypothesis')
plt.xlabel('Input feature', labelpad=10)
plt.ylabel('Target variable', labelpad=10)
plt.title('Linear Regression')
pylab.legend(loc='upper left')
plt.show()

#----


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = np.linspace(-0.2, 2, num=100)
y = np.linspace(-1, 1, num=100)
X, Y = np.meshgrid(x, y)


def compute_j_values(theta0, theta1):
    sume = 0.0
    for i in range(x_size):
        sume = sume + pow((input_y[i] - theta0 - theta1 * input_x[i]), 2)
    return sume/(2 * x_size)


zs = np.array([[compute_j_values(x, y)] for x, y in zip(np.ravel(X), np.ravel(Y))])
Z = zs.reshape(X.shape)

ax.plot_surface(X, Y, Z, cmap=cm.RdBu_r, rstride=1, cstride=1, linewidth=0.9, alpha=0.8)

ax.set_xlabel(r'$\theta_0$', labelpad=10)
ax.set_ylabel(r'$\theta_1$', labelpad=10)
ax.set_zlabel(r'$J(\theta)$', labelpad=10)


T0P = np.asarray(theta0_mesh)
T1P = np.asarray(theta1_mesh)
JP = np.asarray(j_theta_mesh)
a = len(T1P)

if converged:
    for i in range(a):
        ax.plot([T0P[i]], [T1P[i]], [JP[i]], color='black', marker='o', markersize=5.5, linestyle='-')
        plt.pause(time_gap)
else:
    for i in range(50):
        ax.plot([T0P[i]], [T1P[i]], [JP[i]], color='black', marker='o', markersize=5.5, linestyle='-')
        plt.pause(time_gap)

print("over")
plt.show()

x = np.linspace(-0.2, 2, num=100)
y = np.linspace(-1, 1, num=100)
X, Y = np.meshgrid(x, y)
zs = np.array([[compute_j_values(x, y)] for x, y in zip(np.ravel(X), np.ravel(Y))])
Z = zs.reshape(X.shape)
plt.contour(X, Y, Z, 25, color='black')
plt.xlabel(r'$\theta_0$', labelpad=10)
plt.ylabel(r'$\theta_1$', labelpad=10)

if converged:
    for i in range(a):
        plt.plot(T0P[i], T1P[i], color='red', marker='o', markersize=5.5, linestyle='-')
        plt.pause(time_gap)
else:
    for i in range(50):
        plt.plot([T0P[i]], T1P[i], color='red', marker='o', markersize=5.5, linestyle='-')
        plt.pause(time_gap)

print("over")
plt.show()

