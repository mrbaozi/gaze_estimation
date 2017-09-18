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
nodal_point = np.array((0, 0, 0))

# position of light source 1, 2
source1 = np.array((-80, -15, -6))
source2 = np.array((80, -15, -6))

# transform img to camera coordinates
def to_ccs(pt, x_center, y_center, p_pitch, lmb):
    out = []
    for row in pt:
        out.append([p_pitch * (row[0] - x_center), p_pitch * (row[1] - y_center), -lmb])
    return np.array(out)

def b_norm(gi1, gi2, ls1, ls2, o):
    b = np.cross(np.cross(gi1 - o, ls1 - o), \
                 np.cross(ls2 - o, gi2 - o))
    return (b / np.linalg.norm(b))

# for glints 1, 2, pupil center ep = K | R, pt = gi | pi
def k_rp(k_c, pt, bnorm, o, ep):
    num = k_c * np.dot(o - pt, bnorm) \
        - np.sqrt(k_c**2 * np.dot((o - pt)**2, bnorm)**2 \
                  - np.linalg.norm(o - pt)**2 * (k_c**2 - ep**2))
    denom = np.linalg.norm(o - pt)**2
    return (num / denom)

def r_i(k_c, pt, bnorm, o, ep):
    return (o + k_rp(k_c, pt, bnorm, o, ep) * (o - pt))

def c(k_c, bnorm, o):
    return (o + k_c * bnorm)

def p(k_c, pt, bnorm, o, ep):
    return (o + k_rp(k_c, pt, bnorm, o, ep) * (o - pt))

# minimize for k_c
def minimize_this(k_c, pt, bnorm, o, ep, ls):
    lhs = np.dot(ls - r_i(k_c, pt, bnorm, o, ep), r_i(k_c, pt, bnorm, o, ep) \
                 - c(k_c, bnorm, o)) * np.linalg.norm(o - r_i(k_c, pt, bnorm, o, ep))
    rhs = np.dot(o - r_i(k_c, pt, bnorm, o, ep), r_i(k_c, pt, bnorm, o, ep) \
                 - c(k_c, bnorm, o)) * np.linalg.norm(ls - r_i(k_c, pt, bnorm, o, ep))
    return (lhs - rhs)

ppos = np.loadtxt('./data/pupilpos_lefteye.txt')
rpos = np.loadtxt('./data/reflexpos_lefteye.txt')
glint1 = rpos[:, [0, 1]]
glint2 = rpos[:, [2, 3]]
ppos_ccs = to_ccs(ppos, c_x, c_y, pixel_pitch, focal_length)
glint1_ccs = to_ccs(glint1, c_x, c_y, pixel_pitch, focal_length)
glint2_ccs = to_ccs(glint2, c_x, c_y, pixel_pitch, focal_length)

def _main():
    for i in range(len(ppos_ccs)):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(*ppos_ccs[i].T, c='r')
        ax.scatter(*glint1_ccs[i].T, c='g')
        ax.scatter(*glint2_ccs[i].T, c='b')
        plt.grid()
        plt.tight_layout()
        ax.auto_scale_xyz([-0.2, -0.1], [-0.8, -0.6], [-7, -9])
        plt.savefig('./imgs/frame_{:03d}.png'.format(i))
        plt.close()

def main():
    bnorm = b_norm(glint1_ccs[0], glint2_ccs[0], source1, source2, nodal_point)

    # obtain k_c for both glints
    kc1 = optimize.minimize(minimize_this, 0, \
                            args=(glint1_ccs[0], bnorm, nodal_point, R, source1), \
                            tol=1e-4)
    kc2 = optimize.minimize(minimize_this, kc1.x[0], \
                            args=(glint2_ccs[0], bnorm, nodal_point, R, source1), \
                            tol=1e-4)
    print("kc1:\n", kc1, "\nkc2:\n", kc2)

    # use mean as result
    k_c = (kc1.x[0] + kc2.x[0]) / 2
    print("\nk_c:\n", k_c)

    # calculate c and p from kc
    c_res = c(k_c, bnorm, nodal_point)
    p_res = p(k_c, ppos_ccs[0], bnorm, nodal_point, K)
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

    # calculate visual axis g from k_g
    x1 = np.cos(phi_eye + beta_eye) * np.sin(theta_eye + alpha_eye)
    x2 = np.sin(phi_eye + beta_eye)
    x3 = -np.cos(phi_eye + beta_eye) * np.cos(theta_eye + alpha_eye)
    g = c_res + k_g * np.array([x1, x2, x3])
    print("\ng:\n", g)

    # plots
    xmin = -500
    xmax = 500
    x = range(xmin, xmax)
    rng = np.linspace(xmin, xmax, xmax - xmin)
    y1 = np.array([minimize_this(x, glint1_ccs[0], bnorm, nodal_point, R, source1) for x in rng])
    y2 = np.array([minimize_this(x, glint2_ccs[0], bnorm, nodal_point, R, source1) for x in rng])
    plt.plot(x, y1)
    plt.plot(x, y2)
    plt.grid()
    plt.tight_layout()
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(*np.array((source1, source2)).T, c='r')
    ax.scatter(*np.array((glint1_ccs[0], glint2_ccs[0])).T, c='g')
    ax.scatter(*np.array(nodal_point).T, c='k')
    ax.scatter(*ppos_ccs[0].T, c='w')
    ax.scatter(*np.array((c_res, p_res)).T, c='g')
    ax.scatter(*g.T, c='r')
    ax.plot(*np.array((c_res, p_res)).T, c='g')
    plt.grid()
    plt.tight_layout()
    box = 80
    ax.auto_scale_xyz([-box, box], [-box, box], [-box, box])
    plt.show()

if __name__ == '__main__':
    main()
