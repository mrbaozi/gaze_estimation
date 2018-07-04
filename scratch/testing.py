#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import configparser

from modules.PyOSBReceiver import PyOSBReceiver


FMT = '[%(asctime)s.%(msecs)03d] [%(name)s] [%(levelname)s] %(message)s'
DATEFMT = '%Y-%m-%d %H:%M:%S'
logging.basicConfig(format=FMT, datefmt=DATEFMT, level=logging.INFO)


config = configparser.ConfigParser()
config.read('../config/config.ini')

R = config.getfloat('parameters', 'R')
K = config.getfloat('parameters', 'K')
n1 = config.getfloat('parameters', 'n1')
alpha_eye = config.getfloat('parameters', 'alpha_eye')
beta_eye = config.getfloat('parameters', 'beta_eye')

receiver_config = dict(config.items('pyosb_receiver'))
receiver = PyOSBReceiver(receiver_config)

try:
    receiver.run()
except KeyboardInterrupt:
    receiver.stop()
