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
alpha_eye = [-5, 5]
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
source = np.array([[-40, -355, 0],
                   [40, -355, 0]])
source1 = np.array([-40, -355, 0])
source2 = np.array([40, -355, 0])

# ccs to wcs translation vector
t_trans = np.array([0, 0, 0])


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

def k_qsp(k_c, o, uvw, b, rk):
    """k_q, k_s, k_p (2.33, 2.34, 2.35)"""
    num = k_c * np.dot(o - uvw, b) \
        - np.sqrt((k_c * np.dot(o - uvw, b))**2 \
                  - np.linalg.norm(o - uvw)**2 * (k_c**2 - rk**2))
    denom = np.linalg.norm(o - uvw)**2
    return (num / denom)

def glints_qsp(o, uvw, kqsp):
    """pupil center p, reflections s & q (2.10, 2.8, 2.4)"""
    return o + kqsp * (o - uvw)

def curvaturecenter_c(o, b, kc):
    """center of corneal curvature c (2.29)"""
    return o + kc * b

# optimization functions

def solve_kc_qc(kc, u, w, o, l, m, b):
    """substitute q, c and solve for k_c (2.2)"""
    kq = k_qsp(kc, o, u, b, R)
    q = glints_qsp(o, u, kq)
    c = curvaturecenter_c(o, b, kc)

    lhs = np.dot(l - q, q - c) * np.linalg.norm(o - q)
    rhs = np.dot(o - q, q - c) * np.linalg.norm(l - q)
    return lhs - rhs

def solve_kc_sc(kc, u, w, o, l, m, b):
    """substitute s, c and solve for k_c (2.6)"""
    ks = k_qsp(kc, o, w, b, R)
    s = glints_qsp(o, w, ks)
    c = curvaturecenter_c(o, b, kc)

    lhs = np.dot(m - s, s - c) * np.linalg.norm(o - s)
    rhs = np.dot(o - s, s - c) * np.linalg.norm(m - s)
    return lhs - rhs

def find_min_distance(params, eye1, gaze1, eye2, gaze2):
    """Find the minimum distance between left & right visual axes"""
    t1, t2 = params
    vec1 = eye1 + t1 * (gaze1 - eye1)
    vec2 = eye2 + t2 * (gaze2 - eye2)
    return np.linalg.norm(vec1 - vec2)


####
# Main
##

def main(rng, innerplots=False, outerplots=True, savefig=False):
    if rng > 1:
        sys.stdout = DevNull()

    # initialize solution arrays
    glints_wcs = [[] for _ in (0, 1)]
    pupils_wcs = [[] for _ in (0, 1)]
    gazepoints = [[] for _ in (0, 1)]
    mindists = [[] for _ in (0, 1)]
    c_res = [[] for _ in (0, 1)]
    p_res = [[] for _ in (0, 1)]

    # determine coordinate transformation parameters
    kcam = k_cam(phi_cam, theta_cam)
    ic0 = i_cam_0(np.array([0, 1, 0]), kcam)
    jc0 = j_cam_0(kcam, ic0)
    icam = i_cam(ic0, jc0, kappa_cam)
    jcam = j_cam(ic0, jc0, kappa_cam)
    ijkcam = np.array([icam, jcam, kcam])

    # transform gaze targets from pixel to world coordinates
    # TODO

    for i in tqdm(range(rng), ncols=80):  # images to scan
        for j in (0, 1):                  # left, right
            # transform image to camera coordinates
            glint = [to_ccs(glints[j][0][i], c_center, p_pitch),
                     to_ccs(glints[j][1][i], c_center, p_pitch)]
            pupil = to_ccs(ppos[j][i], c_center, p_pitch)

            glint[0] = to_wcs(ijkcam, glint[0], t_trans)
            glint[1] = to_wcs(ijkcam, glint[1], t_trans)
            pupil = to_wcs(ijkcam, pupil, t_trans)

            glints_wcs[j].append([glint[0], glint[1]])
            pupils_wcs[j].append(pupil)

            # bnorm vector
            bnorm = b_norm(glint[0], source[0], source[1], glint[1], nodal_point)

            # obtain k_c for both glints
            args = (glint[0],
                    glint[1],
                    nodal_point,
                    source[0],
                    source[1],
                    bnorm)
            kc1 = optimize.minimize(solve_kc_qc, 0, args=args,
                                      bounds=((-400, 400),),
                                      method='SLSQP', tol=1e-5)
            kc2 = optimize.minimize(solve_kc_sc, kc1.x[0], args=args,
                                      bounds=((-400, 400),),
                                      method='SLSQP', tol=1e-5)

            # use mean as result
            kc = (kc1.x[0] + kc2.x[0]) / 2

            # calculate c and p from kc
            kp = k_qsp(kc, nodal_point, pupil, bnorm, K)
            c_res[j].append(curvaturecenter_c(nodal_point, bnorm, kc))
            p_res[j].append(glints_qsp(nodal_point, pupil, kp))

            # optical axis is vector given by c & p
            o_ax = p_res[j][i] - c_res[j][i]
            o_ax_norm = o_ax / np.linalg.norm(o_ax)

            # calculate phi_eye and theta_eye from c_res and p_res
            phi_eye = np.arcsin((p_res[j][i][1] - c_res[j][i][1]) / K)
            theta_eye = -np.arctan((p_res[j][i][0] - c_res[j][i][0]) /
                                   (p_res[j][i][2] - c_res[j][i][2]))

            # calculate k_g from pan and tilt angles
            k_g = c_res[j][i][2] / (np.cos(phi_eye + beta_eye) * np.cos(theta_eye + alpha_eye[j]))

            # calculate gaze point g from k_g
            x1 = np.cos(phi_eye + beta_eye) * np.sin(np.deg2rad(theta_eye + alpha_eye[j]))
            x2 = np.sin(np.deg2rad(phi_eye + beta_eye))
            x3 = -np.cos(np.deg2rad(phi_eye + beta_eye)) * np.cos(np.deg2rad(theta_eye + alpha_eye[j]))
            gazepoints[j].append(c_res[j][i] + k_g * np.array([x1, x2, x3]))

        # calculate shortest distance between visual axes ("intersection")
        mindist = optimize.minimize(find_min_distance, (1, 1),
                                  args=(c_res[0][i], gazepoints[0][i], c_res[1][i], gazepoints[1][i]),
                                  bounds=((-100, 100),
                                          (-100, 100)),
                                  method='SLSQP',
                                  tol=1e-5)

        for j in (0, 1):
            solp = c_res[j][i] + (gazepoints[j][i] - c_res[j][i]) * mindist.x[j]
            mindists[j].append(solp)

        if innerplots:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(*np.array(glints_wcs[0][i]).T, c='g', label='Glints (left)')
            ax.scatter(*np.array(glints_wcs[1][i]).T, c='y', label='Glints (right)')
            ax.scatter(*pupils_wcs[0][i].T, c='r', label='Pupil center (left)')
            ax.scatter(*pupils_wcs[1][i].T, c='b', label='Pupil center (right)')
            ax.scatter(*nodal_point.T, c='k', label='Nodal point')
            plt.grid()
            ax.auto_scale_xyz([-1, 1], [-1, 1], [0, 8])
            plt.title('In-camera view')
            ax.set_xlabel('x (mm)')
            ax.set_ylabel('y (mm)')
            ax.set_zlabel('z (mm)')
            plt.legend()
            plt.tight_layout()
            if savefig:
                plt.savefig('./plots/in_camera_{:04d}.png'.format(i))
            else:
                plt.show()
            plt.close()

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(*np.array(source).T, c='k')
            ax.scatter(*nodal_point.T, c='k', label='Nodal point')
            ax.scatter(*c_res[0][i].T, c='b', label='Center of corneal curvature (left)')
            ax.scatter(*c_res[1][i].T, c='r', label='Center of corneal curvature (right)')
            ax.scatter(*gazepoints[0][i], c='b', marker='x', label='Gaze point (left)')
            ax.scatter(*gazepoints[1][i], c='r', marker='x', label='Gaze point (right)')
            ax.scatter(*np.array(mindists[0][i]).T, c='y', label='Closest point (left)')
            ax.scatter(*np.array(mindists[1][i]).T, c='y', label='Closest point (right)')
            ax.plot(*np.array((c_res[0][i], gazepoints[0][i])).T, c='b', linestyle='--')
            ax.plot(*np.array((c_res[1][i], gazepoints[1][i])).T, c='r', linestyle='--')
            ax.auto_scale_xyz([-500, 500], [-500, 500], [-1000, 0])
            ax.view_init(elev=110, azim=-90)
            plt.title('Out-of-camera view')
            ax.set_xlabel('x (mm)')
            ax.set_ylabel('y (mm)')
            ax.set_zlabel('z (mm)')
            plt.legend()
            plt.tight_layout()
            if savefig:
                plt.savefig('./plots/out_of_camera_{:04d}.png'.format(i))
            else:
                plt.show()
            plt.close()

    if outerplots:
        mindists[0] = np.array(mindists[0])
        mindists[1] = np.array(mindists[1])
        mindists = mindists[0] + 0.5 * (mindists[1] - mindists[0])
        for j in (0, 1):
            gazepoints[j] = np.array(gazepoints[j])
            gazepoints[j] = gazepoints[j][~np.isnan(gazepoints[j]).any(axis=1)]
            gazepoints[j] = gazepoints[j][abs(gazepoints[j][:, 0]) < 2000]
            gazepoints[j] = gazepoints[j][abs(gazepoints[j][:, 1]) < 2000]
            mindists = mindists[~np.isnan(mindists).any(axis=1)]
            mindists = mindists[abs(mindists[:, 0]) < 1000]
            mindists = mindists[abs(mindists[:, 1]) < 1000]
            mindists = mindists[abs(mindists[:, 2]) < 1000]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(*gazepoints[0].T, label='Gaze (left)')
        ax.scatter(*gazepoints[1].T, label='Gaze (right)')
        plt.title('Gaze points for {} frames'.format(len(gazepoints[0])))
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        ax.set_zlabel('z (mm)')
        plt.legend()
        if savefig:
            plt.savefig('./plots/total_gaze_3d.png')
        else:
            plt.show()
        plt.close()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(*mindists.T, label='Gaze intersect')
        plt.title('Gaze intersect for {} frames'.format(len(gazepoints[0])))
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        ax.set_zlabel('z (mm)')
        plt.legend()
        if savefig:
            plt.savefig('./plots/total_intersect_3d.png')
        else:
            plt.show()
        plt.close()

        fig, ax = plt.subplots()
        ax.scatter([*gazepoints[0].T][0], [*gazepoints[0].T][1], label='Gaze (left)')
        ax.scatter([*gazepoints[1].T][0], [*gazepoints[1].T][1], label='Gaze (right)')
        plt.title('xy projection of gaze points for {} frames'.format(len(gazepoints[0])))
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        plt.legend()
        if savefig:
            plt.savefig('./plots/total_gaze_2d.png')
        else:
            plt.show()
        plt.close()

        fig, ax = plt.subplots()
        ax.scatter([*mindists.T][0], [*mindists.T][1])
        plt.title('xy projection of gaze intersect for {} frames'.format(len(gazepoints[0])))
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        if savefig:
            plt.savefig('./plots/total_intersect_2d.png')
        else:
            plt.show()
        plt.close()


if __name__ == '__main__':
    main(100, innerplots=False, outerplots=True, savefig=False)
