#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Implementation of gaze estimation system proposed by Guestrin et al.
Reference: 'A Novel Head-Free Point-of-Gaze Estimation System'
Equation references (x.xx) in functions correspond to above thesis.
"""


import sys
import numpy as np
import numpy.linalg as la
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm


class GazeMapper(object):
    def __init__(self, args, data):
        # data
        self.data = data

        print(self.data.keys())
        print(self.data['light'].shape)
        print(self.data['reflex'].shape)
        print(self.data['pupil'].shape)
        print(self.data['target'].shape)
        print(self.data['screen_rotation'].shape)

        # eye parameters
        self.eye_K = args.K
        self.eye_R = args.R
        self.eye_alpha = args.alpha
        self.eye_beta = args.beta
        self.n1 = args.n1
        self.n2 = args.n2

        # camera parameters (from calibration)
        self.focal_length = args.cam_focal

        # wcs position of nodal point of camera
        self.nodal_point = np.array([0, 0, self.focal_length])

        # screen plane definition
        self.screenNormal = self.data['screen_rotation'].dot(np.array([0, 0, 1]))
        self.screenPoint = self.data['target'][:, 0]

        # params = (7.8, 4.75, 5, 1.5)
        # bounds = ((6.2, 9.4), (3.8, 5.7), (4, 6), (1, 2))
        # res = minimize(optimize_gaze, params, bounds=bounds,
        #                method='SLSQP', tol=1e-3, options={'maxiter': 1000})
        # print(res)
        # sys.exit(0)

        # calibration parameters (R, K)
        # eye 0: 6.200, 5.365
        # eye 1: TODO not converging :(

        # test gaze center calculation
        intersect0, c, w, w_rot = self.calc_gaze()
        # intersect1, c, w, w_rot = self.calc_gaze(6.2, 5.365, 5.0, 1.5)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(*self.nodal_point.T, c='k', label='Nodal point')
        ax.scatter(*self.data['light'][0].T, c='k', label='Source 1')
        ax.scatter(*self.data['light'][1].T, c='k', label='Source 2')
        ax.scatter(*intersect0.T, c='c', marker='.', linewidth=0.1)
        # ax.scatter(*intersect1.T, c='m', marker='.', linewidth=0.1)
        for i in range(0, len(c), 30):
            ax.plot(*np.array((c[i], c[i] + 400 * w[i])).T,
                    c='b', linestyle='-')
            ax.plot(*np.array((c[i], c[i] + 400 * w_rot[i])).T,
                    c='g', linestyle='-')
        for tgt in np.unique(self.data['target'].T, axis=0):
            ax.scatter(*tgt.T, c='k', marker='x')
        ax.auto_scale_xyz([-300, 300], [0, 300], [1000, 0])
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        ax.set_zlabel('z (mm)')
        plt.tight_layout()
        plt.show()
        plt.close()


    def b_norm(self, l1, l2, u1, u2, o):
        """intersection of planes (3.8)"""
        b = np.cross(np.cross(l1 - o, u1 - o),
                     np.cross(l2 - o, u2 - o))
        return b / la.norm(b)

    def curvaturecenter_c(self, kq, l, u, b, o, r):
        ou_n = (o - u) / la.norm(o - u)
        q = o + kq * ou_n
        oq_n = (o - q) / la.norm(o - q)
        lq_n = (l - q) / la.norm(l - q)
        return(q - r * ((lq_n + oq_n) / la.norm(lq_n + oq_n)))

    def solve_kc_phd1(self, kc, l, u, b, o, r):
        ou_n = (o - u) / la.norm(o - u)
        kq = (kc * np.dot(ou_n, b)
              - np.sqrt(kc**2 * np.dot(ou_n, b)**2 - kc**2 + r**2))
        q = o + kq * ou_n
        oq_n = (o - q) / la.norm(o - q)
        lq_n = (l - q) / la.norm(l - q)
        return np.dot(lq_n - oq_n, q - o + kc * b)

    def solve_kc_phd2(self, kq, l1, l2, u1, u2, b, o, r):
        """minimization problem (3.11)"""
        c1 = self.curvaturecenter_c(kq[0], l1, u1, b, o, r)
        c2 = self.curvaturecenter_c(kq[1], l2, u2, b, o, r)
        return la.norm(c1 - c2)**2

    def to_ccs(self, pt, center, pitch):
        """ics to ccs (2.27)"""
        xy = pitch * (pt - center)
        return np.append(xy, 0)

    def to_wcs(self, ijk_cam, xyz_u, t):
        """ccs to wcs (2.21)"""
        return np.dot(ijk_cam, xyz_u) + t

    def k_cam(self, phi, theta):
        """unit vector k_cam (2.22)"""
        return np.array([np.cos(np.deg2rad(phi)) * np.sin(np.deg2rad(theta)),
                         np.sin(np.deg2rad(phi)),
                         -np.cos(np.deg2rad(phi)) * np.cos(np.deg2rad(theta))])

    def i_cam_0(self, j, k):
        """i_cam_0 (2.23)"""
        return np.cross(j, k) / la.norm(np.cross(j, k))

    def j_cam_0(self, k, ic0):
        """j_cam_0 (2.24)"""
        return np.cross(k, ic0)

    def i_cam(self, ic0, jc0, kappa):
        """i_cam (2.25)"""
        return (np.cos(np.deg2rad(kappa)) * ic0
                + np.sin(np.deg2rad(kappa)) * jc0)

    def j_cam(self, ic0, jc0, kappa):
        """j_cam (2.26)"""
        return (-np.sin(np.deg2rad(kappa)) * ic0
                + np.cos(np.deg2rad(kappa)) * jc0)

    def r_listing(self, phi, theta):
        sp = np.sin(np.deg2rad(phi))
        cp = np.cos(np.deg2rad(phi))
        st = np.sin(np.deg2rad(theta))
        ct = np.cos(np.deg2rad(theta))
        m00 = 1 - (st**2 * cp**2) / (1 + ct * cp)
        m01 = (-sp * st * cp) / (1 + ct * cp)
        m02 = -st * cp
        m10 = (-sp * st * cp) / (1 + ct * cp)
        m11 = (ct * cp + cp**2) / (1 + ct * cp)
        m12 = -sp
        m20 = st * cp
        m21 = sp
        m22 = ct * cp
        return np.array([
            [m00, m01, m02],
            [m10, m11, m12],
            [m20, m21, m22]])

    def theta_eye_pp(self, alpha, beta):
        sa = np.sin(np.deg2rad(alpha))
        ca = np.cos(np.deg2rad(alpha))
        sb = np.sin(np.deg2rad(beta))
        cb = np.cos(np.deg2rad(beta))
        return -np.arctan((sa * cb) / (np.sqrt(ca**2 * cb**2 + sb**2)))

    def phi_eye_pp(self, alpha, beta):
        return -np.arctan(np.tan(np.deg2rad(beta)) / np.cos(np.deg2rad(alpha)))

    def calc_centers(self, pupil, light1, light2, glint1, glint2, node):
        # variable shorthands for notation simplicity (thesis conventions)
        v = pupil
        l1 = light1
        l2 = light2
        u1 = glint1
        u2 = glint2
        o = node

        # bnorm vector
        b = self.b_norm(l1, l2, u1, u2, o)

        # obtain c (center of corneal curvature) from kq (method 2)
        params = (500, 500)
        bounds = ((200, 1000), (200, 1000))
        args = (l1, l2, u1, u2, b, o, self.eye_R)
        kq = minimize(self.solve_kc_phd2, params, args=args, bounds=bounds,
                      method='SLSQP', tol=1e-8, options={'maxiter': 1e5})
        # mean of both results (3.12)
        c1 = self.curvaturecenter_c(kq.x[0], l1, u1, b, o, self.eye_R)
        c2 = self.curvaturecenter_c(kq.x[1], l2, u2, b, o, self.eye_R)
        c = (c1 + c2) / 2

        # kr (3.29)
        ocov = np.dot(o - c, o - v)
        kr = ((-ocov - np.sqrt(ocov**2 - la.norm(o - v)**2
                               * (la.norm(o - c)**2 - self.eye_R**2)))
              / la.norm(o - v)**2)

        # sanity check |r - c| = R (2.42 & 2.43)
        r = o + kr * (o - v)
        np.testing.assert_almost_equal(la.norm(r - c), self.eye_R)

        # (3.32)
        nu = (r - c) / self.eye_R
        # both of (3.31)
        eta0 = (v - o) / la.norm(v - o)
        eta1 = (o - r) / la.norm(o - r)
        # and check if (3.31) holds
        np.testing.assert_almost_equal(eta0, eta1)

        # can use either one but hey...
        eta = (eta0 + eta1) / 2

        # iota from (3.33)
        iota = self.n2 / self.n1 * (
            (np.dot(nu, eta) - np.sqrt((self.n1/self.n2)**2
                                       - 1 + np.dot(nu, eta)**2)) * nu - eta
        )

        # kp (3.37)
        rci = np.dot(r - c, iota)
        kp = -rci - np.sqrt(rci**2 - (self.eye_R**2 - self.eye_K**2))

        # p (3.34)
        p = r + kp * iota

        # |p - c| = K (3.35)
        np.testing.assert_almost_equal(la.norm(p - c), self.eye_K)

        # satisfy constraint (3.36)
        np.testing.assert_array_less(la.norm(p - o), la.norm(c - o))

        return p, c

    def calc_gaze(self):
        p_left, c_left = [], []
        p_right, c_right = [], []
        for index in tqdm(range(self.data['target'].shape[1]), ncols=80):
            # calculate left eye
            eye = 0
            pi, ci = self.calc_centers(
                self.data['pupil'][eye, :, index],
                self.data['light'][0, :],
                self.data['light'][1, :],
                self.data['reflex'][eye, 0, :, index],
                self.data['reflex'][eye, 1, :, index],
                self.nodal_point,
            )
            p_left.append(pi)
            c_left.append(ci)

            # calculate right eye
            eye = 1
            pi, ci = self.calc_centers(
                self.data['pupil'][eye, :, index],
                self.data['light'][0, :],
                self.data['light'][1, :],
                self.data['reflex'][eye, 0, :, index],
                self.data['reflex'][eye, 1, :, index],
                self.nodal_point,
            )
            p_right.append(pi)
            c_right.append(ci)
        p = (np.array(p_left) + np.array(p_right)) / 2
        c = (np.array(c_left) + np.array(c_right)) / 2

        # calculate optic axis and unit vector to targets from curvature center
        w = (p - c) / la.norm(p - c, axis=1)[:, np.newaxis]
        v = (
            (self.data['target'].T - c)
            / la.norm(self.data['target'].T - c, axis=1)[:, np.newaxis]
        )

        # find rotation matrix between optic and target vectors
        R = []
        for wi, vi in zip(w, v):
            n = np.cross(wi, vi)
            sns = la.norm(n)
            cns = np.dot(wi, vi)
            nx = np.array([[0, -n[2], n[1]],
                           [n[2], 0, -n[0]],
                           [-n[1], n[0], 0]])
            Ri = np.identity(3) + nx + nx**2 * (1 - cns) / sns**2
            R.append(Ri)
        R = np.array(R)
        R_mean = np.mean(R, axis=0)

        w_rot = []
        for wi in w:
            w_rot.append(np.dot(R_mean, wi))
        w_rot = np.array(w_rot)

        # TODO implement listing's law
        # phi_pp = phi_eye_pp(alpha_eye, beta_eye)
        # theta_pp = theta_eye_pp(alpha_eye, beta_eye)
        # t_el = -np.arctan(v[:, 0] / v[:, 2])
        # p_el = np.arcsin(v[:, 1])
        # sys.exit(0)

        # find intersection with screen
        intersect = []
        for ci, wi in zip(c, w_rot):
            ndotu = np.dot(self.screenNormal, wi)
            wk = (ci + wi) - self.screenPoint
            si = np.dot(-self.screenNormal, wk) / ndotu
            intersect.append(wk + si * wi + self.screenPoint)
        intersect = np.array(intersect)

        return intersect, c, w, w_rot

    def optimize_gaze(self, params):
        R, K, alpha, beta = params
        intersect, c, w, w_rot = self.calc_gaze(R, K, alpha, beta)
        return np.mean(la.norm(self.data['target'] - intersect, axis=1))**2
