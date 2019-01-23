#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def rotate_axis(theta, axis):
    x, y, z = axis
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    sa, ca = np.sin(np.radians(theta)), np.cos(np.radians(theta))
    R = [[ca + xx * (1 - ca), xy * (1 - ca) - z * sa, xz * (1 - ca) + y * sa],
         [xy * (1 - ca) + z * sa, ca + yy * (1 - ca), yz * (1 - ca) - x * sa],
         [xz * (1 - ca) - y * sa, yz * (1 - ca) + x * sa, ca + zz * (1 - ca)]]
    return np.array(R)


class Preprocessor(object):
    def __init__(self, dataframe, args):
        self.dataFrame = dataframe
        self.cam_center = np.array(args.cam_center)
        self.cam_res = np.array(args.cam_res)
        self.cam_focal = args.cam_focal
        self.cam_pp = args.cam_pp
        self.screen_res = np.array(args.screen_res)
        self.screen_pp = args.screen_pp
        self.screen_center = np.array(args.screen_center)
        self.screen_norm = np.array(args.screen_norm)
        self.light_l = np.array(args.light_l)
        self.light_r = np.array(args.light_r)
        self.preprocessed = False

        rot_ax = np.cross(np.array([0, 0, 1]), self.screen_norm)
        rot_ax /= np.linalg.norm(rot_ax)
        alpha = np.arccos(np.array([0, 0, 1]) @ self.screen_norm)
        self.rot = rotate_axis(np.degrees(alpha), rot_ax)

    def show(self):
        if not self.preprocessed:
            self.preprocess_all()
        fig, ax = plt.subplots(2, 1)
        ax[0].scatter(self.dataFrame['left_eye.pupilpos.x'],
                      self.dataFrame['left_eye.pupilpos.y'])
        ax[0].scatter(self.dataFrame['left_eye.reflexpos.left.x'],
                      self.dataFrame['left_eye.reflexpos.left.y'],
                      marker='.')
        ax[0].scatter(self.dataFrame['left_eye.reflexpos.right.x'],
                      self.dataFrame['left_eye.reflexpos.right.y'],
                      marker='.')
        ax[0].scatter(self.dataFrame['right_eye.pupilpos.x'],
                      self.dataFrame['right_eye.pupilpos.y'])
        ax[0].scatter(self.dataFrame['right_eye.reflexpos.left.x'],
                      self.dataFrame['right_eye.reflexpos.left.y'],
                      marker='.')
        ax[0].scatter(self.dataFrame['right_eye.reflexpos.right.x'],
                      self.dataFrame['right_eye.reflexpos.right.y'],
                      marker='.')
        ax[0].set_title("Camera sensor")
        ax[0].set_xlabel("x (mm)")
        ax[0].set_ylabel("y (mm)")
        ax[1].scatter(self.dataFrame['gaze_target.x'],
                      self.dataFrame['gaze_target.y'])
        ax[1].scatter(self.dataFrame['gaze_point.x'],
                      self.dataFrame['gaze_point.y'],
                      marker='x')
        ax[1].scatter(self.dataFrame['left_eye.gazepos.x'],
                      self.dataFrame['left_eye.gazepos.y'],
                      marker='.')
        ax[1].scatter(self.dataFrame['right_eye.gazepos.x'],
                      self.dataFrame['right_eye.gazepos.y'],
                      marker='.')
        ax[1].set_title("Screen")
        ax[1].set_xlabel("x (mm)")
        ax[1].set_ylabel("y (mm)")
        plt.tight_layout()
        plt.show()


    def preprocess_all(self):
        self.fix_reflex_positions()
        for key in self.dataFrame.keys():
            if "gaze" in key:
                # convert gaze from [0, 1] to pixels
                self.rescale_to_screen(key)
                if ".x" in key:
                    # flip left/right
                    self.dataFrame[key] = self.screen_res[0] - self.dataFrame[key]
                    # recenter origin (middle of screen is zero)
                    self.dataFrame[key] -= self.screen_res[0] / 2
                if ".y" in key:
                    # flip up/down
                    self.dataFrame[key] = self.screen_res[1] - self.dataFrame[key]
                # convert pixels to mm
                self.dataFrame[key] *= self.screen_pp
            if any(s in key for s in ["reflex", "pupil", "eyecoords"]):
                if ".x" in key:
                    # recenter coordinates to cam midpoint (from calibration)
                    self.dataFrame[key] -= self.cam_center[0]
                if ".y" in key:
                    # recenter coordinates to cam midpoint (from calibration)
                    self.dataFrame[key] -= self.cam_center[1]
                # convert cam pixels to mm
                self.dataFrame[key] *= self.cam_pp

        # calculate offsets w.r.t. screen center
        self.light_l += np.array(
            [0, self.screen_res[1] * self.screen_pp / 2, 0])
        self.light_r += np.array(
            [0, self.screen_res[1] * self.screen_pp / 2, 0])

        # rotate everything in screen coordinates to world coordinates
        self.light_l = self.rot @ self.light_l + self.screen_center
        self.light_r = self.rot @ self.light_r + self.screen_center

        # self.dataFrame['gaze_target.y'] += self.screen_center[1]

        # and rotate lights and gaze targets to world coordinates
        targets = np.stack([self.dataFrame['gaze_target.x'],
                            self.dataFrame['gaze_target.y'],
                            np.zeros(self.dataFrame.shape[0])],
                           axis=1)
        # TODO
        # print(np.dot(targets, self.rot.T).T)
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # xr, yr, zr = targets.T
        # ax.scatter(xr, yr, zr)
        xr, yr, zr = (np.dot(targets, self.rot.T) + self.screen_center).T
        # ax.scatter(xr, yr, zr)
        # xr, yr, zr = np.dot(targets, self.rot.T).T
        # ax.scatter(xr, yr, zr)
        # ax.scatter(0, 0, 0)
        # ax.scatter(*self.light_l)
        # ax.scatter(*self.light_r)
        # plt.show()
        # sys.exit()
        self.dataFrame['gaze_target.x'] = xr
        self.dataFrame['gaze_target.y'] = yr
        self.dataFrame['gaze_target.z'] = zr

        # add wcs z-axis values for 2d data (fill zeros for camera images)
        self.dataFrame['left_eye.pupilpos.z'] = np.zeros(self.dataFrame.shape[0])
        self.dataFrame['right_eye.pupilpos.z'] = np.zeros(self.dataFrame.shape[0])
        self.dataFrame['left_eye.reflexpos.left.z'] = np.zeros(self.dataFrame.shape[0])
        self.dataFrame['left_eye.reflexpos.right.z'] = np.zeros(self.dataFrame.shape[0])
        self.dataFrame['right_eye.reflexpos.left.z'] = np.zeros(self.dataFrame.shape[0])
        self.dataFrame['right_eye.reflexpos.right.z'] = np.zeros(self.dataFrame.shape[0])

        # drop all values where we don't have pupils and reflexes
        # actually this also drops everything that doesn't already have a gaze
        # --> maybe fix that (not important for current recordings)
        self.dataFrame = self.dataFrame[(self.dataFrame['left_eye.complete']) &
                                        (self.dataFrame['right_eye.complete'])]

        # we are done, set flag
        self.preprocessed = True

    def fliplr(self):
        for eye in ['left_eye', 'right_eye']:
            self.dataFrame[eye + '.pupilpos.x'] = \
                self.cam_res[0] - self.dataFrame[eye + '.pupilpos.x']
            self.dataFrame[eye + '.reflex_center.x'] = \
                self.cam_res[0] - self.dataFrame[eye + '.reflex_center.x']
            self.dataFrame[eye + '.reflexpos.left.x'] = \
                self.cam_res[0] - self.dataFrame[eye + '.reflexpos.left.x']
            self.dataFrame[eye + '.reflexpos.right.x'] = \
                self.cam_res[0] - self.dataFrame[eye + '.reflexpos.right.x']

    def flipud(self):
        for eye in ['left_eye', 'right_eye']:
            self.dataFrame[eye + '.pupilpos.y'] = \
                self.cam_res[1] - self.dataFrame[eye + '.pupilpos.y']
            self.dataFrame[eye + '.reflex_center.y'] = \
                self.cam_res[1] - self.dataFrame[eye + '.reflex_center.y']
            self.dataFrame[eye + '.reflexpos.left.y'] = \
                self.cam_res[1] - self.dataFrame[eye + '.reflexpos.left.y']
            self.dataFrame[eye + '.reflexpos.right.y'] = \
                self.cam_res[1] - self.dataFrame[eye + '.reflexpos.right.y']

    def get_wcs_data(self):
        '''Output all data that is needed for gaze estimation'''
        if not self.preprocessed:
            self.preprocess_all()
        data = {}
        data['light'] = np.array([self.light_l, self.light_r])
        data['pupil'] = np.array(
            [[self.dataFrame['left_eye.pupilpos.x'],
              self.dataFrame['left_eye.pupilpos.y'],
              self.dataFrame['left_eye.pupilpos.z']],
             [self.dataFrame['right_eye.pupilpos.x'],
              self.dataFrame['right_eye.pupilpos.y'],
              self.dataFrame['right_eye.pupilpos.z']]])
        data['reflex'] = np.array(
            [[[self.dataFrame['left_eye.reflexpos.left.x'],
               self.dataFrame['left_eye.reflexpos.left.y'],
               self.dataFrame['left_eye.reflexpos.left.z']],
              [self.dataFrame['left_eye.reflexpos.right.x'],
               self.dataFrame['left_eye.reflexpos.right.y'],
               self.dataFrame['left_eye.reflexpos.left.z']]],
             [[self.dataFrame['right_eye.reflexpos.left.x'],
               self.dataFrame['right_eye.reflexpos.left.y'],
               self.dataFrame['right_eye.reflexpos.left.z']],
              [self.dataFrame['right_eye.reflexpos.right.x'],
               self.dataFrame['right_eye.reflexpos.right.y'],
               self.dataFrame['right_eye.reflexpos.left.z']]]])
        data['target'] = np.array(
            [self.dataFrame['gaze_target.x'],
             self.dataFrame['gaze_target.y'],
             self.dataFrame['gaze_target.z']])
        data['screen_rotation'] = self.rot
        data['screen_normal'] = self.screen_norm
        data['screen_center'] = self.screen_center
        return data

    def rescale_to_screen(self, key, screen_x=None, screen_y=None):
        if screen_x is None:
            screen_x = self.screen_res[0]
        if screen_y is None:
            screen_y = self.screen_res[1]
        if ".x" == key[-2:]:
            self.dataFrame[key] *= screen_x
        if ".y" == key[-2:]:
            self.dataFrame[key] *= screen_y

    def fix_reflex_positions(self):
        for eye in ['left_eye', 'right_eye']:
            for xy in ['x', 'y']:
                eyecoord = self.dataFrame[eye + '.eyecoords.' + xy]
                self.dataFrame[eye + '.pupilpos.' + xy] += eyecoord
                self.dataFrame[eye + '.reflex_center.' + xy] += eyecoord
                self.dataFrame[eye + '.reflexpos.left.' + xy] += eyecoord
                self.dataFrame[eye + '.reflexpos.right.' + xy] += eyecoord

    def get_dataframe(self):
        return self.dataFrame
