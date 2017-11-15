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
K = 4.75
R = 7.8
alpha_eye = [5, -5]
beta_eye = 1.5
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
theta_cam = 0.0
kappa_cam = 0.0

# position of nodal point of camera
nodal_point = np.array([0, 0, focal_length])

# position of light source 1, 2
source = np.array([[-110, -390, 0],
                   [110, -390, 0]])

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
targets = 0.282 * targets - np.array([0.282 * 860, 0.282 * 1050 + 36])
# targets = 0.282 * targets - np.array([0.282 * 860, -36])

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

def main(glints, pupils):
    # variable shorthands for notation simplicity
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

    # # obtain k_c for both glints (method 1)
    # args = (l[0], u[0], b, o, R)
    # kc1 = fsolve(solve_kc_phd1, 0, args=args)
    # args = (l[1], u[1], b, o, R)
    # kc2 = fsolve(solve_kc_phd1, 0, args=args)
    # kc = (kc1[0] + kc2[0]) / 2
    # c = o + kc * b

    # obtain c from kq (method 2)
    params = (500, 500)
    bounds = ((1, 1000), (1, 1000))
    args = (l[0], l[1], u[0], u[1], b, o, R)
    kq = minimize(solve_kc_phd2, params, args=args, bounds=bounds, method='SLSQP')
    c1 = curvaturecenter_c(kq.x[0], l[0], u[0], b, o, R)
    c2 = curvaturecenter_c(kq.x[1], l[1], u[1], b, o, R)
    c = (c1 + c2) / 2

    # calculate coordinate of pupil center
    ocov = np.dot(o - c, o - v)
    kr = (-ocov - np.sqrt(ocov**2 - LA.norm(o - v)**2 * (LA.norm(o - c)**2 - R**2))) / LA.norm(o - v)**2
    r = o + kr * (o - v)
    nu = (r - c) / R
    eta = (v - o) / LA.norm(v - o)
    iota = n2/n1 * ((np.dot(nu, eta) - np.sqrt((n1/n2)**2 - 1 + (np.dot(nu, eta)**2)) * nu) - eta)
    rci = np.dot(r - c, iota)
    kp = -rci - np.sqrt(rci**2 - R**2 - K**2)
    p = r + kp * iota

    # optical axis w defined by c and p
    w = (p - c) / LA.norm(p - c)

    #TODO
    #calculate visual axis by rotating optical axis
    return w


if __name__ == '__main__':

    w = []
    for i in tqdm(range(len(targets)), ncols=80):
        w.append(main([glints[0][0][i], glints[0][1][i]], ppos[0][i]))
    w = np.array(w)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(w.T[0], w.T[1])
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    plt.tight_layout()
    plt.show()
