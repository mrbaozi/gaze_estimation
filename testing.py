#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import configparser
import numpy as np
import cv2
import flycapture2 as fc2

config = configparser.ConfigParser()
config.read('./config/config.ini')

R = config.getfloat('parameters', 'R')
K = config.getfloat('parameters', 'K')
n1 = config.getfloat('parameters', 'n1')
alpha_eyes = config.getfloat('parameters', 'alpha_eyes ')
beta_eyes = config.getfloat('parameters', 'beta_eyes ')

c = fc2.Context()
c.connect(*c.get_camera_from_index(0))
im = fc2.Image()
c.start_capture()
c.retrieve_buffer(im)
c.stop_capture()
c.disconnect()
frame = np.array(im)
