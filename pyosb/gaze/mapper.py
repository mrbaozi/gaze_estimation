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

        # eye parameters
        self.eye_K = args.eye_K
        self.eye_R = args.eye_R
        self.eye_alpha = args.eye_alpha
        self.eye_beta = args.eye_beta
        self.n1 = args.n1
        self.n2 = args.n2

        # camera parameters (from calibration)
        self.focal_length = args.cam_focal

        # wcs position of nodal point of camera
        self.nodal_point = np.array([0, 0, self.focal_length])

        # screen plane definition
        self.screen_normal = np.array(args.screen_norm)
        self.screen_point = args.screen_center

        # calibration or gaze calculation
        self.is_calibration = False
        self.last_objective = 0
        self.iterations = 0

    def calibrate(self, x0=None, bounds=None):
        targets = self.data['target'].T
        last_target = targets[0]
        unique_targets = [last_target, ]
        pupils = []
        reflex_left = []
        reflex_right = []
        pupil_mean = []
        reflex_left_mean = []
        reflex_right_mean = []
        for i, target in enumerate(targets):
            if np.array_equal(target, last_target):
                pupils.append(self.data['pupil'][0].T[i])
                reflex_left.append(self.data['reflex'][0, 0].T[i])
                reflex_right.append(self.data['reflex'][0, 1].T[i])
            else:
                pupil_mean.append(np.mean(pupils, axis=0))
                reflex_left_mean.append(np.mean(reflex_left, axis=0))
                reflex_right_mean.append(np.mean(reflex_right, axis=0))
                pupils.clear()
                reflex_left.clear()
                reflex_right.clear()
                unique_targets.append(np.array(target))
                last_target = target
        pupil_mean.append(np.mean(pupils, axis=0))
        reflex_left_mean.append(np.mean(reflex_left, axis=0))
        reflex_right_mean.append(np.mean(reflex_right, axis=0))
        pupils.clear()
        reflex_left.clear()
        reflex_right.clear()
        unique_targets = np.array(unique_targets).T
        pupil_mean = np.array(pupil_mean).T
        reflex_left_mean = np.array(reflex_left_mean).T
        reflex_right_mean = np.array(reflex_right_mean).T

        if x0 is None:
            x0 = (self.eye_R, self.eye_K, self.eye_alpha[0], self.eye_beta)
        if bounds is None:
            # bounds = ((6.2, 9.4), (3.8, 5.7), (4, 6), (1, 2))
            bounds = ((5, 10), (2, 7), (-6, 6), (-3, 3))
        self.is_calibration = True
        res = minimize(self.optimize_gaze, x0=x0,
                       args=(unique_targets, pupil_mean,
                             reflex_left_mean, reflex_right_mean,
                             self.explicit_refraction),
                       bounds=bounds, method='L-BFGS-B',
                       tol=1e-6, options={'maxiter': 2000, 'disp': False})
        self.is_calibration = False
        self.iterations = 0
        self.last_objective = 0
        print(res)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(*self.nodal_point.T, marker='o', c='k', label='Nodal point')
        ax.scatter(*self.data['light'][0].T, marker='v', c='k', label='Source 1')
        ax.scatter(*self.data['light'][1].T, marker='^', c='k', label='Source 2')

        # calculate gaze point for each eye
        w, p, c, v, g1 = self.single_gaze(pupil_mean,
                                          reflex_left_mean, reflex_right_mean,
                                          self.explicit_refraction,
                                          *res.x)
        _, _, _, _, g2 = self.single_gaze(pupil_mean,
                                          reflex_left_mean, reflex_right_mean,
                                          self.explicit_refraction,
                                          self.eye_R, self.eye_K,
                                          self.eye_alpha[0], self.eye_beta)
        ax.scatter(*g1.T, c='c', marker='.', linewidth=2.0, label='Optimized gaze')
        ax.scatter(*g2.T, c='r', marker='x', linewidth=2.0, label='Initial gaze')
        for i in range(0, len(c)):
            # ax.plot(*np.array((c[i], c[i] + 400 * w[i])).T,
            #         c='b', linestyle='-')
            # ax.plot(*np.array((c[i], c[i] + 400 * v[i])).T,
            #         c='g', linestyle='-')
            for tgt in np.unique(self.data['target'].T, axis=0):
                ax.scatter(*tgt.T, c='k', marker='x')

        ax.auto_scale_xyz([-400, 400], [0, 400], [400, 0])
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        ax.set_zlabel('z (mm)')
        ax.legend()
        plt.tight_layout()
        plt.show()
        plt.close()

    @staticmethod
    def b_norm(l1, l2, u1, u2, o):
        """intersection of planes (3.8)"""
        b = np.cross(np.cross(l1 - o, u1 - o),
                     np.cross(l2 - o, u2 - o))
        return b / la.norm(b)

    @staticmethod
    def curvaturecenter_c(kq, l, u, b, o, r):
        ou_n = (o - u) / la.norm(o - u)
        q = o + kq * ou_n
        oq_n = (o - q) / la.norm(o - q)
        lq_n = (l - q) / la.norm(l - q)
        return(q - r * ((lq_n + oq_n) / la.norm(lq_n + oq_n)))

    @staticmethod
    def solve_kc_phd1(kc, l, u, b, o, r):
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

    @staticmethod
    def implicit_refraction(pupil, node, curvature_center, R, K, n1, n2):
        """Pupil center calculation with implicit refraction model (3.3.3)"""
        # Note: R unused but function call should match explicit_refraction()

        # variable shorthands for notation simplicity (thesis conventions)
        v = pupil
        o = node
        c = curvature_center

        # kp (3.45)
        ocov = np.dot(o - c, o - v)
        kp = ((-ocov - np.sqrt(ocov**2 - la.norm(o - v)**2
                               * (la.norm(o - c)**2 - K**2)))
              / la.norm(o - v)**2)

        # pupil center p (3.42)
        p = o + kp * (o - v)

        # sanity checks (3.43 & 3.44)
        np.testing.assert_almost_equal(la.norm(p - c), K)
        np.testing.assert_array_less(la.norm(p - o), la.norm(c - o))

        return p

    @staticmethod
    def explicit_refraction(pupil, node, curvature_center, R, K, n1, n2):
        """Pupil center calculation with explicit refraction model (3.3.2)"""
        # variable shorthands for notation simplicity (thesis conventions)
        v = pupil
        o = node
        c = curvature_center

        # kr (3.29)
        ocov = np.dot(o - c, o - v)
        kr = ((-ocov - np.sqrt(ocov**2 - la.norm(o - v)**2
                               * (la.norm(o - c)**2 - R**2)))
              / la.norm(o - v)**2)

        # sanity check |r - c| = R (2.42 & 2.43)
        r = o + kr * (o - v)
        np.testing.assert_almost_equal(la.norm(r - c), R)

        # (3.32)
        nu = (r - c) / R
        # both of (3.31)
        eta0 = (v - o) / la.norm(v - o)
        eta1 = (o - r) / la.norm(o - r)
        # and check if (3.31) holds
        np.testing.assert_almost_equal(eta0, eta1)

        # can use either one but hey...
        eta = (eta0 + eta1) / 2

        # iota from (3.33)
        iota = n2 / n1 * (
            (np.dot(nu, eta) - np.sqrt((n1 / n2)**2
                                       - 1 + np.dot(nu, eta)**2)) * nu - eta
        )

        # kp (3.37)
        rci = np.dot(r - c, iota)
        kp = -rci - np.sqrt(rci**2 - (R**2 - K**2))

        # p (3.34)
        p = r + kp * iota

        # |p - c| = K (3.35)
        np.testing.assert_almost_equal(la.norm(p - c), K)

        # satisfy constraint (3.36)
        np.testing.assert_array_less(la.norm(p - o), la.norm(c - o))

        return p

    def calc_curvature_center(self, light1, light2, glint1, glint2, node, R):
        # variable shorthands for notation simplicity (thesis conventions)
        l1 = light1
        l2 = light2
        u1 = glint1
        u2 = glint2
        o = node

        # bnorm vector
        b = self.b_norm(l1, l2, u1, u2, o)

        # obtain c (center of corneal curvature) from kq (method 2)
        params = np.array((500, 500))
        bounds = ((200, 1000), (200, 1000))
        args = (l1, l2, u1, u2, b, o, R)
        kq = minimize(self.solve_kc_phd2, params, args=args, bounds=bounds,
                      method='SLSQP', tol=1e-8, options={'maxiter': 1e5})
        # mean of both results (3.12)
        c1 = self.curvaturecenter_c(kq.x[0], l1, u1, b, o, R)
        c2 = self.curvaturecenter_c(kq.x[1], l2, u2, b, o, R)
        return (c1 + c2) / 2

    def optical_axis(self, R, K, pupil, glint1, glint2,
                     refraction_model, interval):
        """Calculate optical axis of eye (Section 3.3)"""

        v = pupil
        u1 = glint1
        u2 = glint2

        p = []
        c = []
        for idx in tqdm(range(0, pupil.shape[1], interval),
                        ncols=80, disable=self.is_calibration):
            ci = self.calc_curvature_center(
                self.data['light'][0, :],
                self.data['light'][1, :],
                u1[:, idx],
                u2[:, idx],
                self.nodal_point, R)
            pi = refraction_model(v[:, idx], self.nodal_point, ci, R, K,
                                  self.n1, self.n2)
            p.append(pi)
            c.append(ci)
        p = np.array(p)
        c = np.array(c)

        # unit vector w in direction of optic axis (3.38)
        w = (p - c) / la.norm(p - c, axis=1)[:, np.newaxis]

        return w, p, c

    @staticmethod
    def visual_axis(w, eye_alpha, eye_beta, interval):
        """Reconstruct visual axis (Section 3.4)"""

        # convert eye_alpha and eye_beta to radians
        alpha = np.radians(eye_alpha)
        beta = np.radians(eye_beta)

        # determine eye pan and tilt angles from w (A.19 & A.20)
        eye_theta = -np.arctan(w[:, 0] / w[:, 2])
        eye_phi = np.arcsin(w[:, 1])

        v = []

        # alternative method to get v from paper
        v = np.array([
            np.cos(eye_phi + beta) * np.sin(eye_theta + alpha),
            np.sin(eye_phi + beta),
            -np.cos(eye_phi + beta) * np.cos(eye_theta + alpha)
        ]).T

        return v

    @staticmethod
    def scene_intersect(n, p, c, v):
        # kg for intersection with scene (3.61)
        # basically, kg is the z value at which g intersects the screen
        g = []
        d = np.dot(n, p)
        for ci, vi in zip(c, v):
            kg = ((d - np.dot(n, ci))
                  / np.dot(n, vi))

            # gaze point g = c + kg * v (2.31)
            g.append(ci + kg * vi)
        g = np.array(g)

        return g

    def single_gaze(self,
                    v, u1, u2,
                    refraction_model,
                    R, K, eye_alpha, eye_beta,
                    interval=1):
        w, p, c = self.optical_axis(R, K, v, u1, u2,
                                    refraction_model, interval)
        v = self.visual_axis(w, eye_alpha, eye_beta, interval)
        g = self.scene_intersect(self.screen_normal, self.screen_point, c, v)
        return w, p, c, v, g

    def calc_gaze(self, R=None, K=None,
                  eye_alpha=None, eye_beta=None,
                  eye='both', interval=1, refraction_type='explicit',
                  show=False):
        """Gets gaze intersect with screen from optical and visual axis"""

        if R is None:
            R = self.eye_R
        if K is None:
            K = self.eye_K

        if eye == 'both':
            eye_idx = [0, 1]
        elif eye == 'left':
            eye_idx = [0]
        elif eye == 'right':
            eye_idx = [1]
        else:
            print(f"Unrecognized eye index: {eye}")
            sys.exit(1)

        if eye_alpha is None:
            eye_alpha = self.eye_alpha
        if eye_beta is None:
            eye_beta = self.eye_beta

        if refraction_type == 'explicit':
            refraction_model = self.explicit_refraction
        elif refraction_type == 'implicit':
            refraction_model = self.implicit_refraction
        else:
            print(f"Unrecognized refraction model: {refraction_type}")
            sys.exit(1)

        if show:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(*self.nodal_point.T, c='k', label='Nodal point')
            ax.scatter(*self.data['light'][0].T, c='k', label='Source 1')
            ax.scatter(*self.data['light'][1].T, c='k', label='Source 2')

        # calculate gaze point for each eye
        g = []
        for i in eye_idx:
            v = self.data['pupil'][i]
            u1 = self.data['reflex'][i, 0]
            u2 = self.data['reflex'][i, 1]
            w, p, c, v, gi = self.single_gaze(v, u1, u2, refraction_model,
                                              R, K, eye_alpha[i], eye_beta,
                                              interval)
            g.append(gi)

            if show:
                ax.scatter(*gi.T, c='c', marker='.', linewidth=0.1)
                for i in range(0, len(c), 30):
                    ax.plot(*np.array((c[i], c[i] + 400 * w[i])).T,
                            c='b', linestyle='-')
                    ax.plot(*np.array((c[i], c[i] + 400 * v[i])).T,
                            c='g', linestyle='-')
                for tgt in np.unique(self.data['target'].T, axis=0):
                    ax.scatter(*tgt.T, c='k', marker='x')

        g = np.array(g)

        if show:
            ax.auto_scale_xyz([-300, 300], [0, 300], [1000, 0])
            ax.set_xlabel('x (mm)')
            ax.set_ylabel('y (mm)')
            ax.set_zlabel('z (mm)')
            plt.tight_layout()
            plt.show()
            plt.close()

        return np.mean(g, axis=0) if g.shape[0] == 2 else g[0]

    def optimize_gaze(self, x0, t, v, u1, u2, refraction_model):
        R, K, eye_alpha, eye_beta = x0
        w, p, c, v, g = self.single_gaze(v, u1, u2, refraction_model,
                                         R, K, eye_alpha, eye_beta)

        objective = la.norm(np.mean(np.absolute(t.T - g), axis=0))

        self.iterations += 1

        print(f"Iteration {self.iterations:3d}  -  "
              f"Objective: {objective:12.8f}  |  "
              f"delta = {objective - self.last_objective:12.8f}")

        self.last_objective = objective

        return objective
