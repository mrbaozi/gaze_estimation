#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def rotation_matrix(theta, axis):
    x, y, z = axis
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    sa, ca = np.sin(np.radians(theta)), np.cos(np.radians(theta))
    R = [[ca + xx * (1 - ca), xy * (1 - ca) - z * sa, xz * (1 - ca) + y * sa],
         [xy * (1 - ca) + z * sa, ca + yy * (1 - ca), yz * (1 - ca) - x * sa],
         [xz * (1 - ca) - y * sa, yz * (1 - ca) + x * sa, ca + zz * (1 - ca)]]
    return np.array(R)


R = rotation_matrix(30, [1, 0, 0])
n_points = 2

node_ccs = np.array([0, 0, 8])
node_wcs = np.dot(node_ccs, R)  # inverse rotation (R*x != x*R)

_x = np.linspace(-50, 50, n_points)
_y = np.linspace(0, 100, n_points) + 38
_z = np.zeros(n_points) - 8

points = np.stack([_x, _y, _z], axis=1)
_xr, _yr, _zr = np.dot(points, R.T).T

x, y = np.meshgrid(_x, _y)
z = np.stack([_z] * n_points, axis=1)

xr, yr = np.meshgrid(_xr, _yr)
zr = np.stack([_zr] * n_points, axis=1)

fig = plt.figure()
# ax = fig.add_subplot(111, aspect='equal', projection='3d')
ax = fig.add_subplot(111, aspect='equal')

ax.plot(y, z)
ax.plot(yr, zr)
ax.scatter(0, 0)
ax.scatter(*node_ccs[1:])
ax.scatter(*node_wcs[1:])
ax.set_xlabel('y')
ax.set_ylabel('z')

# ax.plot_surface(x, y, z)
# ax.plot_surface(xr, yr, zr)
# ax.scatter(0, 0, 0)
# ax.scatter(*node_ccs)
# ax.scatter(*node_wcs)
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')

plt.show()
