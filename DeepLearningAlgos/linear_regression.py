from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
#%matplotlib inline
N = 50

#simple 2D linear regression example
x = np.linspace(0, 10, num=N).reshape(N, 1)
y = np.linspace(0, 10, num=N).reshape(N, 1)
t = np.random.normal(0, 1, (N, 1))
y += t
beta_hat = inv(x.T.dot(x)).dot(x.T).dot(y)

fig, ax = plt.subplots()
ax.plot(x, y, 'bo')
ax.plot(x, x.dot(beta_hat), 'r')
plt.show()

z = np.linspace(10, 20, num=N).reshape(N, 1)
x_3d = np.concatenate((x, z), axis=1)

beta_hat_3d = inv(x_3d.T.dot(x_3d)).dot(x_3d.T).dot(y)

fig_3d = plt.figure()
ax_3d = fig_3d.add_subplot(111, projection='3d')
ax_3d.scatter(x, z, y)
y_3d_hat = x_3d.dot(beta_hat_3d)
ax_3d.plot(x, z, y_3d_hat.reshape(N), color='red')
plt.show()
