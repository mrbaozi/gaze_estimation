#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# eye parameters
K = 4.2
R = 7.2
alpha_eye = 5
beta_eye = 1.5

# camera parameters (from calibration)
focal_length = 8.42800210895 # in mm
nodal_x = 2.6996358692163716 # in mm
nodal_y = 2.2439755534153347 # in mm
c_x = 1.07985435e+03         # in px
c_y = 8.97590221e+02         # in px
f_x = 3.37120084e+03         # in px
f_y = 3.37462371e+03         # in px
pixel_pitch = 0.0025         # 2.5 micro meter pixel size in mm

# position of nodal point of camera
o = np.array((nodal_x, nodal_y, focal_length))

# position of light source 1, 2
ls = np.array(((-80, -15, 0), (80, -15, 0)))

# position of glint image on sensor plane 1, 2
gi = np.array(((-0.664, -1.03725, 0), (-0.6795, -1.03775, 0)))

# position of pupil image on sensor plane 1, 2
pi = np.array((-0.68275, -1.0585, 0))

b = np.cross(np.cross(gi[0] - o, ls[0] - o), np.cross(ls[1] - o, gi[1] - o))
b_norm = b / np.linalg.norm(b)

# transform img to camera coordinates
def to_ccs((x, y, z), x_center, y_center, p_pitch, lmb):
    return ((p_pitch * (x - x_center), p_pitch * (y - y_center), -lmb * z))

# for glints 1, 2
def k_r(k_c, i):
    num = k_c * np.dot(o - gi[i], b_norm) \
        - np.sqrt(k_c**2 * np.dot((o - gi[i])**2, b_norm)**2 \
                  - np.linalg.norm(o - gi[i])**2 * (k_c**2 - R**2))
    denom = np.linalg.norm(o - gi[i])**2
    return (num / denom)

# for pupil center
def k_p(k_c):
    num = k_c * np.dot(o - pi, b_norm) \
        - np.sqrt(k_c**2 * np.dot((o - pi)**2, b_norm)**2 \
                  - np.linalg.norm(o - pi)**2 * (k_c**2 - K**2))
    denom = np.linalg.norm(o - pi)**2
    return (num / denom)

def r_i(k_c, i):
    return (o + k_r(k_c, i) * (o - gi[i]))

def c(k_c):
    return (o + k_c * b_norm)

def p(k_c):
    return (o + k_p(k_c) * (o - pi))

# minimize for k_c
def minimize_this(k_c, i):
    lhs = np.dot(ls[i] - r_i(k_c, i), r_i(k_c, i) - c(k_c)) * np.linalg.norm(o - r_i(k_c, i))
    rhs = np.dot(o - r_i(k_c, i), r_i(k_c, i) - c(k_c)) * np.linalg.norm(ls[i] - r_i(k_c, i))
    return (lhs - rhs)

def main():
    # obtain k_c for both glints
    kc1 = optimize.minimize(lambda x: minimize_this(x, 0), 0, tol=1e-4)
    kc2 = optimize.minimize(lambda x: minimize_this(x, 1), kc1.x[0], tol=1e-4)
    print("kc1:\n", kc1, "\nkc2:\n", kc2)

    # use mean as result
    k_c = (kc1.x[0] + kc2.x[0]) / 2
    print("\nk_c:\n", k_c)

    # calculate c and p from kc
    c_res = c(k_c)
    p_res = p(k_c)
    print("\nc_res:\n", c_res, "\np_res:\n", p_res)

    # optic axis is vector given by c & p
    o_ax = p_res - c_res
    o_ax_norm = o_ax / np.linalg.norm(o_ax)
    print("\noptical axis (norm):\n", np.linalg.norm(o_ax))
    print("\noptical axis (unit):\n", o_ax_norm)

    # calculate phi_eye and theta_eye from c_res and p_res
    phi_eye = np.arcsin((p_res[1] - c_res[1]) / K)
    theta_eye = -np.arctan((p_res[0] - c_res[0]) / (p_res[2] - c_res[2]))
    print("\nphi_eye:\n", phi_eye, "\ntheta_eye:\n", theta_eye)

    # calculate k_g from pan and tilt angles
    k_g = c_res[2] / (np.cos(phi_eye + beta_eye) * np.cos(theta_eye + alpha_eye))
    print("\nk_g:\n", k_g)

    # # plots
    # xmin = -500
    # xmax = 500
    # x = range(xmin, xmax)
    # rng = np.linspace(xmin, xmax, xmax - xmin)
    # y1 = np.array([minimize_this(x, 0) for x in rng])
    # y2 = np.array([minimize_this(x, 1) for x in rng])
    # plt.plot(x, y1)
    # plt.plot(x, y2)
    # plt.grid()
    # plt.tight_layout()
    # plt.show()

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter((ls[0, 0], ls[1, 0]), (ls[0, 1], ls[1, 1]), (ls[0, 2], ls[1, 2]), c='r')
    # ax.scatter((gi[0, 0], gi[1, 0]), (gi[0, 1], gi[1, 1]), (gi[0, 2], gi[1, 2]), c='k')
    # ax.scatter(o[0], o[1], o[2], c='b')
    # ax.scatter(pi[0], pi[1], pi[2], c='k')
    # ax.scatter(c_res[0], c_res[1], c_res[2], c='g')
    # ax.scatter(p_res[0], p_res[1], p_res[2], c='g')
    # ax.plot([p_res[0], c_res[0]], [p_res[1], c_res[1]], [p_res[2], c_res[2]], c='g')
    # plt.grid()
    # plt.tight_layout()
    # box = 40
    # ax.auto_scale_xyz([-box, box], [-box, box], [-box, box])
    # plt.show()

if __name__ == '__main__':
    main()
