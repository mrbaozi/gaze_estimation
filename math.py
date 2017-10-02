#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Implementation of gaze estimation system proposed by Guestrin et al.
Reference: 'A Novel Head-Free Point-of-Gaze Estimation System'
Equation references (x.xx) in functions correspond to above thesis.
"""


import sys
import numpy as np
from scipy import optimize
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
alpha_eye_l = -5
alpha_eye_r = 5
beta_eye = 1.5

# camera parameters (from calibration)
focal_length = 8.42800210895 # in mm
nodal_x = 2.6996358692163716 # in mm
nodal_y = 2.2439755534153347 # in mm
c_center = np.array([1.07985435e+03, 8.97590221e+02])
f_x = 3.37120084e+03         # in px
f_y = 3.37462371e+03         # in px
p_pitch = 0.0025         # 2.5 micro meter pixel size in mm
phi_cam = 0.0
theta_cam = 0.0
kappa_cam = 0.0

# position of nodal point of camera
nodal_point = np.array([0, 0, focal_length])

# position of light source 1, 2
source1 = np.array([-40, -355, 0])
source2 = np.array([40, -355, 0])


####
# Data points
##

ppos_l = np.loadtxt('./data/pupilpos_lefteye.txt')
ppos_r = np.loadtxt('./data/pupilpos_righteye.txt')
rpos_l = np.loadtxt('./data/reflexpos_lefteye.txt')
rpos_r = np.loadtxt('./data/reflexpos_righteye.txt')
glints1_l = rpos_l[:, [0, 1]]
glints2_l = rpos_l[:, [2, 3]]
glints1_r = rpos_r[:, [0, 1]]
glints2_r = rpos_r[:, [2, 3]]


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
                     np.cos(np.deg2rad(phi)) * np.cos(np.deg2rad(theta))])

def i_cam_0(j, k):
    """i_cam_0 (2.23)"""
    return np.cross(j, k) / np.linalg.norm(np.cross(j, k))

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

def b_norm(u, l, m, w, o):
    """intersection of planes (2.28)"""
    b = np.cross(np.cross(u - o, l - o), \
                 np.cross(m - o, w - o))
    return b / np.linalg.norm(b)

def k_q(k_c, o, u, b, r):
    """k_q (2.33)"""
    num = k_c * np.dot(o - u, b) \
        - np.sqrt((k_c * np.dot(o - u, b))**2 \
                  - np.linalg.norm(o - u)**2 * (k_c**2 - r**2))
    denom = np.linalg.norm(o - u)**2
    return (num / denom)

def k_s(k_c, o, w, b, r):
    """k_s (2.34)"""
    num = k_c * np.dot(o - w, b) \
        - np.sqrt((k_c * np.dot(o - w, b))**2 \
                  - np.linalg.norm(o - w)**2 * (k_c**2 - r**2))
    denom = np.linalg.norm(o - w)**2
    return (num / denom)

def k_p(k_c, o, v, b, k):
    """k_p (2.35)"""
    num = k_c * np.dot(o - v, b) \
        - np.sqrt((k_c * np.dot(o - v, b))**2 \
                  - np.linalg.norm(o - v)**2 * (k_c**2 - k**2))
    denom = np.linalg.norm(o - v)**2
    return (num / denom)

def pupilcenter_p(o, v, kp):
    """pupil center p (2.10)"""
    return o + kp * (o - v)

def reflection_s(o, w, ks):
    """reflection s (2.8)"""
    return o + ks * (o - w)

def reflection_q(o, u, kq):
    """reflection q (2.4)"""
    return o + kq * (o - u)

def curvaturecenter_c(o, b, kc):
    """center of corneal curvature c (2.29)"""
    return o + kc * b

def solution1(kc, u, w, o, l, m, b):
    """substitute q, c and solve for k_c (2.2)"""
    kq = k_q(kc, o, u, b, R)
    q = reflection_q(o, u, kq)
    c = curvaturecenter_c(o, b, kc)

    lhs = np.dot(l - q, q - c) * np.linalg.norm(o - q)
    rhs = np.dot(o - q, q - c) * np.linalg.norm(l - q)
    return lhs - rhs

def solution2(kc, u, w, o, l, m, b):
    """substitute s, c and solve for k_c (2.6)"""
    ks = k_s(kc, o, w, b, R)
    s = reflection_s(o, w, ks)
    c = curvaturecenter_c(o, b, kc)

    lhs = np.dot(m - s, s - c) * np.linalg.norm(o - s)
    rhs = np.dot(o - s, s - c) * np.linalg.norm(m - s)
    return lhs - rhs

def find_min_distance(params, eye1, gaze1, eye2, gaze2):
    t1, t2 = params
    vec1 = eye1 + t1 * (gaze1 - eye1)
    vec2 = eye2 + t2 * (gaze2 - eye2)
    return np.linalg.norm(vec1 - vec2)


def main(rng, it=False):
    gazepoints_l = []
    gazepoints_r = []
    mindist_l = []
    mindist_r = []
    if it:
        sys.stdout = DevNull()
    for i in tqdm(range(rng)):
        # transform image to camera coordinates
        glint1_l = to_ccs(glints1_l[i], c_center, p_pitch)
        glint2_l = to_ccs(glints2_l[i], c_center, p_pitch)
        glint1_r = to_ccs(glints1_r[i], c_center, p_pitch)
        glint2_r = to_ccs(glints2_r[i], c_center, p_pitch)
        pupil_l = to_ccs(ppos_l[i], c_center, p_pitch)
        pupil_r = to_ccs(ppos_r[i], c_center, p_pitch)
        print("Left Glint 1 (ccs): {}".format(glint1_l))
        print("Left Glint 2 (ccs): {}".format(glint2_l))
        print("Right Glint 1 (ccs): {}".format(glint1_r))
        print("Right Glint 2 (ccs): {}".format(glint2_r))
        print("Left Pupil (ccs): {}".format(pupil_l))
        print("Right Pupil (ccs): {}".format(pupil_r))

        # determine coordinate transformation parameters
        kcam = k_cam(phi_cam, theta_cam)
        ic0 = i_cam_0(np.array([0, 1, 0]), kcam)
        jc0 = j_cam_0(kcam, ic0)
        icam = i_cam(ic0, jc0, kappa_cam)
        jcam = j_cam(ic0, jc0, kappa_cam)
        ijkcam = np.array([icam, jcam, kcam])
        print("Transformation matrix ijk_cam:\n{}".format(ijkcam))

        # transform camera to world coordinates
        t = np.array([0, 0, 0])
        glint1_l = to_wcs(ijkcam, glint1_l, t)
        glint2_l = to_wcs(ijkcam, glint2_l, t)
        glint1_r = to_wcs(ijkcam, glint1_r, t)
        glint2_r = to_wcs(ijkcam, glint2_r, t)
        pupil_l = to_wcs(ijkcam, pupil_l, t)
        pupil_r = to_wcs(ijkcam, pupil_r, t)
        print("Left Glint 1 (wcs): {}".format(glint1_l))
        print("Left Glint 2 (wcs): {}".format(glint2_l))
        print("Right Glint 1 (wcs): {}".format(glint1_r))
        print("Right Glint 2 (wcs): {}".format(glint2_r))
        print("Left Pupil (wcs): {}".format(pupil_l))
        print("Right Pupil (wcs): {}".format(pupil_r))

        # priors
        ul = glint1_l
        wl = glint2_l
        ur = glint1_r
        wr = glint2_r
        vl = pupil_l
        vr = pupil_r
        o = nodal_point
        l = source1
        m = source2
        bl = b_norm(ul, l, m, wl, o)
        br = b_norm(ur, l, m, wr, o)

        # obtain k_c for both glints
        kc1_l = optimize.minimize(solution1, 0,
                                  args=(ul, wl, o, l, m, bl),
                                  bounds=((-400, 400),),
                                  method='SLSQP',
                                  tol=1e-5)
        kc2_l = optimize.minimize(solution2, kc1_l.x[0],
                                  args=(ul, wl, o, l, m, bl),
                                  bounds=((-400, 400),),
                                  method='SLSQP',
                                  tol=1e-5)
        kc1_r = optimize.minimize(solution1, 0,
                                  args=(ur, wr, o, l, m, br),
                                  bounds=((-400, 400),),
                                  method='SLSQP',
                                  tol=1e-5)
        kc2_r = optimize.minimize(solution2, kc1_r.x[0],
                                  args=(ur, wr, o, l, m, br),
                                  bounds=((-400, 400),),
                                  method='SLSQP',
                                  tol=1e-5)
        print("\nSolution 1 (left):\n{}".format(kc1_l))
        print("\nSolution 2 (left):\n{}\n".format(kc2_l))
        print("\nSolution 1 (right):\n{}".format(kc1_r))
        print("\nSolution 2 (right):\n{}\n".format(kc2_r))

        # use mean as result
        kc_l = (kc1_l.x[0] + kc2_l.x[0]) / 2
        kc_r = (kc1_r.x[0] + kc2_r.x[0]) / 2
        print("k_c (left): {}".format(kc_l))
        print("k_c (right): {}".format(kc_r))

        # calculate c and p from kc
        kp_l = k_p(kc_l, o, vl, bl, K)
        kp_r = k_p(kc_r, o, vr, br, K)
        c_res_l = curvaturecenter_c(o, bl, kc_l)
        p_res_l = pupilcenter_p(o, vl, kp_l)
        c_res_r = curvaturecenter_c(o, br, kc_r)
        p_res_r = pupilcenter_p(o, vr, kp_r)
        print("Center of corneal curvature (left): {}".format(c_res_l))
        print("Pupil center (left): {}".format(p_res_l))
        print("Center of corneal curvature (right): {}".format(c_res_r))
        print("Pupil center (right): {}".format(p_res_r))

        # optic axis is vector given by c & p
        o_ax_l = p_res_l - c_res_l
        o_ax_r = p_res_r - c_res_r
        o_ax_norm_l = o_ax_l / np.linalg.norm(o_ax_l)
        o_ax_norm_r = o_ax_r / np.linalg.norm(o_ax_r)
        print("Optical axis (left): {}".format(o_ax_l))
        print("Optical axis norm (left): {}".format(np.linalg.norm(o_ax_l)))
        print("Optical axis unit (left): {}".format(o_ax_norm_l))
        print("Optical axis (right): {}".format(o_ax_r))
        print("Optical axis norm (right): {}".format(np.linalg.norm(o_ax_r)))
        print("Optical axis unit (right): {}".format(o_ax_norm_r))

        # calculate phi_eye and theta_eye from c_res and p_res
        phi_eye_l = np.arcsin((p_res_l[1] - c_res_l[1]) / K)
        theta_eye_l = -np.arctan((p_res_l[0] - c_res_l[0]) / (p_res_l[2] - c_res_l[2]))
        phi_eye_r = np.arcsin((p_res_r[1] - c_res_r[1]) / K)
        theta_eye_r = -np.arctan((p_res_r[0] - c_res_r[0]) / (p_res_r[2] - c_res_r[2]))
        print("Eye tilt phi_eye (left): {}".format(phi_eye_l))
        print("Eye pan theta_eye (left): {}".format(theta_eye_l))
        print("Eye tilt phi_eye (right): {}".format(phi_eye_r))
        print("Eye pan theta_eye (right): {}".format(theta_eye_r))

        # calculate k_g from pan and tilt angles
        k_g_r = c_res_l[2] / (np.cos(phi_eye_l + beta_eye) * np.cos(theta_eye_l + alpha_eye_l))
        k_g_l = c_res_r[2] / (np.cos(phi_eye_r + beta_eye) * np.cos(theta_eye_r + alpha_eye_r))
        print("k_g (left): {}".format(k_g_l))
        print("k_g (right): {}".format(k_g_r))

        # calculate gaze point g from k_g
        x1_l = np.cos(phi_eye_l + beta_eye) * np.sin(np.deg2rad(theta_eye_l + alpha_eye_l))
        x2_l = np.sin(np.deg2rad(phi_eye_l + beta_eye))
        x3_l = -np.cos(np.deg2rad(phi_eye_l + beta_eye)) * np.cos(np.deg2rad(theta_eye_l + alpha_eye_l))
        gazepoint_l = c_res_l + k_g_l * np.array([x1_l, x2_l, x3_l])
        x1_r = np.cos(phi_eye_r + beta_eye) * np.sin(np.deg2rad(theta_eye_r + alpha_eye_r))
        x2_r = np.sin(np.deg2rad(phi_eye_r + beta_eye))
        x3_r = -np.cos(np.deg2rad(phi_eye_l + beta_eye)) * np.cos(np.deg2rad(theta_eye_r + alpha_eye_r))
        gazepoint_r = c_res_r + k_g_r * np.array([x1_r, x2_r, x3_r])

        gazepoints_l.append(gazepoint_l)
        gazepoints_r.append(gazepoint_r)

        print("Gaze point (left): {}".format(gazepoint_l))
        print("Gaze point (right): {}".format(gazepoint_r))

        # calculate shortest distance between visual axes ("intersection")
        mindist = optimize.minimize(find_min_distance, (1, 1),
                                  args=(c_res_l, gazepoint_l, c_res_r, gazepoint_r),
                                  bounds=((-100, 100),
                                          (-100, 100)),
                                  method='SLSQP',
                                  tol=1e-5)
        solp1 = c_res_l + (gazepoint_l - c_res_l) * mindist.x[0]
        solp2 = c_res_r + (gazepoint_r - c_res_r) * mindist.x[1]
        mindist_l.append(solp1)
        mindist_r.append(solp2)
        print(mindist)


        # plots
        if not it:
            xmin = -500
            xmax = 500
            x = range(xmin, xmax)
            rng = np.linspace(xmin, xmax, xmax - xmin)
            y1l = np.array([solution1(x, ul, wl, o, l, m, bl) for x in rng])
            y2l = np.array([solution2(x, ul, wl, o, l, m, bl) for x in rng])
            y1r = np.array([solution1(x, ur, wr, o, l, m, br) for x in rng])
            y2r = np.array([solution2(x, ur, wr, o, l, m, br) for x in rng])
            plt.plot(x, y1l)
            plt.plot(x, y2l)
            plt.plot(x, y1r)
            plt.plot(x, y2r)
            plt.grid()
            plt.tight_layout()
            plt.show()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(*np.array((l, m)).T, c='k')
        ax.scatter(*np.array((ul, wl)).T, c='g')
        ax.scatter(*np.array((ur, wr)).T, c='g')
        ax.scatter(*vl.T, c='g')
        ax.scatter(*vr.T, c='g')
        ax.scatter(*o.T, c='k')
        ax.scatter(*np.array((c_res_l, p_res_l, gazepoint_l)).T, c='b')
        ax.scatter(*np.array((c_res_r, p_res_r, gazepoint_r)).T, c='r')
        ax.scatter(*np.array((solp1, solp2)).T, c='y')
        ax.plot(*np.array((c_res_l, gazepoint_l)).T, c='b')
        ax.plot(*np.array((c_res_r, gazepoint_r)).T, c='r')
        plt.grid()
        plt.tight_layout()
        box = 500
        ax.auto_scale_xyz([-box, box], [-box, box], [-box, box])
        ax.view_init(elev=110, azim=-90)
        if it:
            plt.savefig('./plots/frame_{:04d}.png'.format(i))
        else:
            plt.show()
        plt.close()

    # gazepoints_l = np.array(gazepoints_l)
    # gazepoints_r = np.array(gazepoints_r)
    # gazepoints_l = gazepoints_l[~np.isnan(gazepoints_l).any(axis=1)]
    # gazepoints_r = gazepoints_r[~np.isnan(gazepoints_r).any(axis=1)]
    # gazepoints_l = gazepoints_l[abs(gazepoints_l[:, 0]) < 2000]
    # gazepoints_r = gazepoints_r[abs(gazepoints_r[:, 0]) < 2000]
    # gazepoints_l = gazepoints_l[abs(gazepoints_l[:, 1]) < 2000]
    # gazepoints_r = gazepoints_r[abs(gazepoints_r[:, 1]) < 2000]
    # gazepoints_l = gazepoints_l[abs(gazepoints_l[:, 2]) < 2000]
    # gazepoints_r = gazepoints_r[abs(gazepoints_r[:, 2]) < 2000]
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(*gazepoints_l.T)
    # ax.scatter(*gazepoints_r.T)
    # plt.show()

    # mindist_l = np.array(mindist_l)
    # mindist_r = np.array(mindist_r)
    # mindist = mindist_l + 0.5 * (mindist_r - mindist_l)
    # mindist = mindist[~np.isnan(mindist).any(axis=1)]
    # mindist = mindist[abs(mindist[:, 0]) < 1000]
    # mindist = mindist[abs(mindist[:, 1]) < 1000]
    # mindist = mindist[abs(mindist[:, 2]) < 1000]
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(*mindist.T)
    # plt.show()


if __name__ == '__main__':
    rng = 1
    main(rng, 0)
