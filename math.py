#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Implementation of gaze estimation system proposed by Guestrin et al.
Reference: 'A Novel Head-Free Point-of-Gaze Estimation System'
Equation references (x.xx) in functions correspond to above thesis.
"""


import sys
import numpy as np
import numpy.linalg as LA
from scipy.optimize import minimize, fsolve
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm


####
# Utility
##

class DevNull(object):
    """ For deactivating print output (yes, it's hacky af) """
    def write(self, arg):
        pass
    def flush(self):
        pass


####
# System parameters / priors
##

# eye parameters
# eye_K = 4.2
# eye_R = 8.2
# alpha_eye = [5, -5]
eye_alpha = 5
eye_beta = 1.5
n1 = 1.3375
n2 = 1.0

# camera parameters (from calibration)
focal_length = 8.42800210895 # in mm
nodal_x = 2.6996358692163716 # in mm
nodal_y = 2.2439755534153347 # in mm
c_center = np.array([1.07985435e+03, 8.97590221e+02])
f_x = 3.37120084e+03         # in px
f_y = 3.37462371e+03         # in px
p_pitch = 0.0025         # 2.5 micro meter pixel size in mm
phi_cam = -17.0
# phi_cam = 0.0
theta_cam = 0.0
kappa_cam = 0.0

# position of nodal point of camera
nodal_point = np.array([0, 0, focal_length])

# position of light source 1, 2
source = np.array([[-110, -390, 0],
                   [110, -390, 0]])

# screen plane definition
screenNormal = np.array([0, 0, 1])
screenPoint = np.array([0, 0, 0])

# ccs to wcs translation vector
t_trans = np.array([0, 0, 0])


####
# Data points
##

ppos_l = np.loadtxt('./data/pupilpos_lefteye.txt')
ppos_r = np.loadtxt('./data/pupilpos_righteye.txt')
rpos_l = np.loadtxt('./data/reflexpos_lefteye.txt')
rpos_r = np.loadtxt('./data/reflexpos_righteye.txt')
targets = np.loadtxt('./data/targets.txt')

# transform gaze targets from pixel to world coordinates
#TODO target coordinates are not quite right
targets = 0.282 * targets - np.array([0.282 * 860, 0.282 * 1050 + 36])
targets = np.insert(targets, 2, 0, axis=1)

glints = [[rpos_l[:, [0, 1]], rpos_l[:, [2, 3]]],
          [rpos_r[:, [0, 1]], rpos_r[:, [2, 3]]]]
ppos = [ppos_l, ppos_r]


####
# Coordinate transforms (2.5 & 2.6)
##

def to_ccs(pt, center, pitch):
    """ics to ccs (2.27)"""
    xy = pitch * (pt - center)
    return np.append(xy, 0)

def to_wcs(ijk_cam, xyz_u, t):
    """ccs to wcs (2.21)"""
    return np.dot(ijk_cam, xyz_u) + t

def k_cam(phi, theta):
    """unit vector k_cam (2.22)"""
    return np.array([np.cos(np.deg2rad(phi)) * np.sin(np.deg2rad(theta)),
                     np.sin(np.deg2rad(phi)),
                     -np.cos(np.deg2rad(phi)) * np.cos(np.deg2rad(theta))])

def i_cam_0(j, k):
    """i_cam_0 (2.23)"""
    return np.cross(j, k) / LA.norm(np.cross(j, k))

def j_cam_0(k, ic0):
    """j_cam_0 (2.24)"""
    return np.cross(k, ic0)

def i_cam(ic0, jc0, kappa):
    """i_cam (2.25)"""
    return np.cos(np.deg2rad(kappa)) * ic0 + np.sin(np.deg2rad(kappa)) * jc0

def j_cam(ic0, jc0, kappa):
    """j_cam (2.26)"""
    return -np.sin(np.deg2rad(kappa)) * ic0 + np.cos(np.deg2rad(kappa)) * jc0


####
# Gaze estimation (2.7)
##

def b_norm(l1, l2, u1, u2, o):
    """intersection of planes (2.28)"""
    b = np.cross(np.cross(l1 - o, u1 - o), \
                 np.cross(l2 - o, u2 - o))
    return b / LA.norm(b)

def curvaturecenter_c(kq, l, u, b, o, r):
    ou_n = (o - u) / LA.norm(o - u)
    q = o + kq * ou_n
    oq_n = (o - q) / LA.norm(o - q)
    lq_n = (l - q) / LA.norm(l - q)
    return(q - r * ((lq_n + oq_n) / LA.norm(lq_n + oq_n)))

def solve_kc_phd1(kc, l, u, b, o, r):
    ou_n = (o - u) / LA.norm(o - u)
    kq = kc * np.dot(ou_n, b) - np.sqrt(kc**2 * np.dot(ou_n, b)**2 - kc**2 + r**2)
    q = o + kq * ou_n
    oq_n = (o - q) / LA.norm(o - q)
    lq_n = (l - q) / LA.norm(l - q)
    return np.dot(lq_n - oq_n, q - o + kc * b)

def solve_kc_phd2(kq, l1, l2, u1, u2, b, o, r):
    c1 = curvaturecenter_c(kq[0], l1, u1, b, o, r)
    c2 = curvaturecenter_c(kq[1], l2, u2, b, o, r)
    return LA.norm(c1 - c2)**2


####
# Main loops
##

def calc_centers(glints, pupils, eye_R, eye_K):
    # variable shorthands for notation simplicity (thesis conventions)
    o = nodal_point
    l = source

    # determine coordinate transformation parameters
    kcam = k_cam(phi_cam, theta_cam)
    ic0 = i_cam_0(np.array([0, 1, 0]), kcam)
    jc0 = j_cam_0(kcam, ic0)
    icam = i_cam(ic0, jc0, kappa_cam)
    jcam = j_cam(ic0, jc0, kappa_cam)
    ijkcam = np.array([icam, jcam, kcam])

    # transform image to camera coordinates
    u = [to_ccs(glints[0], c_center, p_pitch),
         to_ccs(glints[1], c_center, p_pitch)]
    v = to_ccs(pupils, c_center, p_pitch)
    u[0] = to_wcs(ijkcam, u[0], t_trans)
    u[1] = to_wcs(ijkcam, u[1], t_trans)
    v = to_wcs(ijkcam, v, t_trans)

    # bnorm vector
    b = b_norm(l[0], l[1], u[0], u[1], o)

    # obtain c from kq (method 2)
    params = (500, 500)
    bounds = ((400, 600), (400, 600))
    args = (l[0], l[1], u[0], u[1], b, o, eye_R)
    kq = minimize(solve_kc_phd2, params, args=args, bounds=bounds,
                  method='SLSQP', tol=1e-3, options={'maxiter': 1000})
    c1 = curvaturecenter_c(kq.x[0], l[0], u[0], b, o, eye_R)
    c2 = curvaturecenter_c(kq.x[1], l[1], u[1], b, o, eye_R)
    c = (c1 + c2) / 2

    # calculate coordinate of pupil center
    ocov = np.dot(o - c, o - v)
    kr = (-ocov - np.sqrt(ocov**2 - LA.norm(o - v)**2 * (LA.norm(o - c)**2 - eye_R**2))) / LA.norm(o - v)**2
    r = o + kr * (o - v)
    nu = (r - c) / eye_R
    eta = (v - o) / LA.norm(v - o)
    iota = n2/n1 * ((np.dot(nu, eta) - np.sqrt((n1/n2)**2 - 1 + (np.dot(nu, eta)**2)) * nu) - eta)
    rci = np.dot(r - c, iota)
    kp = -rci - np.sqrt(rci**2 - eye_R**2 - eye_K**2)
    p = r + kp * iota

    return p, c

def calc_gaze(eye_R, eye_K):
    p, c = [], []
    for i in tqdm(range(len(targets)), ncols=80):
        # pi, ci = calc_centers([glints[0][0][i], glints[0][1][i]], ppos[0][i], eye_R, eye_K)
        pi, ci = calc_centers([glints[1][0][i], glints[1][1][i]], ppos[1][i], eye_R, eye_K)
        p.append(pi)
        c.append(ci)
    p = np.array(p)
    c = np.array(c)

    # calculate optic axis and unit vector to targets from curvature center
    w = (p - c) / LA.norm(p - c, axis=1)[:, np.newaxis]
    v = (targets - c) / LA.norm(targets - c, axis=1)[:, np.newaxis]

    # find rotation matrix between optic and target vectors
    R = []
    for wi, vi in zip(w, v):
        n = np.cross(wi, vi)
        sns = LA.norm(n)
        cns = np.dot(wi, vi)
        nx = np.array([[0, -n[2], n[1]],
                       [n[2], 0, -n[0]],
                       [-n[1], n[0], 0]])
        Ri = np.identity(3) + nx + nx**2 * (1 - cns) / sns**2
        R.append(Ri)
    R = np.array(R)
    R_mean = np.mean(R, axis=0)

    w_rot = []
    for wi, Ri in zip(w, R):
        w_rot.append(np.dot(R_mean, wi))
        # w_rot.append(np.dot(Ri, wi))
    w_rot = np.array(w_rot)

    # find intersection with screen
    intersect = []
    for ci, wi in zip(c, w_rot):
        ndotu = np.dot(screenNormal, wi)
        wk = (ci + wi) - screenPoint
        si = np.dot(-screenNormal, wk) / ndotu
        intersect.append(wk + si * wi + screenPoint)
    intersect = np.array(intersect)

    return intersect, c, w, w_rot

def optimize_gaze(params):
    R, K = params
    intersect, c, w, w_rot = calc_gaze(R, K)
    return np.mean(LA.norm(targets - intersect, axis=1))**2

if __name__ == '__main__':

    # params = (8, 4)
    # bounds = ((4, 12), (1, 7))
    # res = minimize(optimize_gaze, params, bounds=bounds,
    #                method='SLSQP', tol=1e-1, options={'maxiter': 1000})
    # print(res)

    # calibration parameters (R, K)
    # eye 0: 9.25, 1.13
    # eye 1: not converging :(

    intersect0, c, w, w_rot = calc_gaze(8.2, 4.2)
    intersect1, c, w, w_rot = calc_gaze(9.25, 1.13)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(*nodal_point.T, c='k', label='Nodal point')
    ax.scatter(*source[0].T, c='k', label='Source 1')
    ax.scatter(*source[1].T, c='k', label='Source 2')
    ax.scatter(*intersect0.T, c='c', marker='.', linewidth=0.1)
    ax.scatter(*intersect1.T, c='m', marker='.', linewidth=0.1)
    for i in range(0, len(c), 30):
        ax.plot(*np.array((c[i], c[i] + 400 * w[i])).T, c='b', linestyle='-')
        ax.plot(*np.array((c[i], c[i] + 400 * w_rot[i])).T, c='g', linestyle='-')
    for tgt in np.unique(targets, axis=0):
        ax.scatter(*tgt.T, c='k', marker='x')
    ax.auto_scale_xyz([-400, 400], [-400, 200], [500, 0])
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    plt.tight_layout()
    plt.show()
    plt.close()
