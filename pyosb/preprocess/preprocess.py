#!/usr/bin/env python3
# -*- coding: utf-8 -*-


class Preprocessor(object):
    def __init__(self, dataframe, args):
        self.cam_x = args.cam_x
        self.cam_y = args.cam_y
        self.screen_x = args.screen_x
        self.screen_y = args.screen_y
        self.dataFrame = dataframe

    def rescale_to_screen(self, key, screen_x=None, screen_y=None):
        if screen_x is None:
            screen_x = self.screen_x
        if screen_y is None:
            screen_y = self.screen_y
        self.dataFrame[key + '.x'] *= screen_x
        self.dataFrame[key + '.y'] *= screen_y

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
