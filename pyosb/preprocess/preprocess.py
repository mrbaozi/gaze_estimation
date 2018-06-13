#!/usr/bin/env python3
# -*- coding: utf-8 -*-


class Preprocessor(object):
    def __init__(self, dataframe, args):
        self.cam_x = args.cam_x
        self.cam_y = args.cam_y
        self.cam_pp = args.cam_pp
        self.screen_x = args.screen_x
        self.screen_y = args.screen_y
        self.screen_pp = args.screen_pp
        self.dataFrame = dataframe

    def preprocess_all(self):
        self.fix_reflex_positions()
        for key in self.dataFrame.keys():
            if "gaze" in key:
                self.rescale_to_screen(key)
                self.dataFrame[key] *= self.screen_pp
            if any(s in key for s in ["reflex", "pupil", "eyecoords"]):
                self.dataFrame[key] *= self.cam_pp
        # self.screen_to_wcs('gaze_target')
        # self.screen_to_wcs('gaze_point')
        # self.screen_to_wcs('left_eye.gazepos')
        # self.screen_to_wcs('right_eye.gazepos')
        # self.fliplr()
        # self.flipud()

    def rescale_to_screen(self, key, screen_x=None, screen_y=None):
        if screen_x is None:
            screen_x = self.screen_x
        if screen_y is None:
            screen_y = self.screen_y
        if ".x" == key[-2:]:
            self.dataFrame[key] *= screen_x
        if ".y" == key[-2:]:
            self.dataFrame[key] *= screen_y

    def screen_to_ccs(self, key, screen_x=None, screen_y=None, screen_pp=None):
        if screen_x is None:
            screen_x = self.screen_x
        if screen_y is None:
            screen_y = self.screen_y
        if screen_pp is None:
            screen_pp = self.screen_pp
        # for xy in ['x', 'y']:
        #     self.dataFrame[key + '.ccs.' + xy] *= screen_pp
        #     self.dataFrame[key + '.ccs.' + xy] -= screen_pp * 

    def rescale_to_cam(self, key, cam_x=None, cam_y=None):
        if cam_x is None:
            cam_x = self.cam_x
        if cam_y is None:
            cam_y = self.cam_y
        self.dataFrame[key + '.x'] *= cam_x
        self.dataFrame[key + '.y'] *= cam_y

    def fix_reflex_positions(self):
        for eye in ['left_eye', 'right_eye']:
            for xy in ['x', 'y']:
                eyecoord = self.dataFrame[eye + '.eyecoords.' + xy]
                self.dataFrame[eye + '.pupilpos.' + xy] += eyecoord
                self.dataFrame[eye + '.reflex_center.' + xy] += eyecoord
                self.dataFrame[eye + '.reflexpos.left.' + xy] += eyecoord
                self.dataFrame[eye + '.reflexpos.right.' + xy] += eyecoord

    def fliplr(self):
        for eye in ['left_eye', 'right_eye']:
            self.dataFrame[eye + '.pupilpos.x'] = \
                self.cam_x - self.dataFrame[eye + '.pupilpos.x']
            self.dataFrame[eye + '.reflex_center.x'] = \
                self.cam_x - self.dataFrame[eye + '.reflex_center.x']
            self.dataFrame[eye + '.reflexpos.left.x'] = \
                self.cam_x - self.dataFrame[eye + '.reflexpos.left.x']
            self.dataFrame[eye + '.reflexpos.right.x'] = \
                self.cam_x - self.dataFrame[eye + '.reflexpos.right.x']

    def flipud(self):
        for eye in ['left_eye', 'right_eye']:
            self.dataFrame[eye + '.pupilpos.y'] = \
                self.cam_y - self.dataFrame[eye + '.pupilpos.y']
            self.dataFrame[eye + '.reflex_center.y'] = \
                self.cam_y - self.dataFrame[eye + '.reflex_center.y']
            self.dataFrame[eye + '.reflexpos.left.y'] = \
                self.cam_y - self.dataFrame[eye + '.reflexpos.left.y']
            self.dataFrame[eye + '.reflexpos.right.y'] = \
                self.cam_y - self.dataFrame[eye + '.reflexpos.right.y']

    def get_dataframe(self):
        return self.dataFrame
