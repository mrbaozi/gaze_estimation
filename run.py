#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import configargparse
import matplotlib.pyplot as plt
from pyosb.fileio.eyeosbparser import EyeInfoParser
from pyosb.preprocess.preprocess import Preprocessor
from pyosb.gaze.mapper import GazeMapper


def main(options):
    # create pandas dataframe from json recordings and do some preprocessing
    parser = EyeInfoParser(options)
    pp = Preprocessor(parser.get_dataframe(), options)
    # pp.show()
    data = pp.get_wcs_data()

    mapper = GazeMapper(options, data)

    # mapper.calc_gaze(eye='left', refraction_type='explicit', show=True)
    mapper.calibrate()

    # _, cl, wl, _ = mapper.calc_gaze(eye='left', refraction_type=refraction)
    # _, cr, wr, _ = mapper.calc_gaze(eye='right', refraction_type=refraction)

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(*cl.T, c='g', marker='.', label='cl')
    # ax.scatter(*cr.T, c='r', marker='.', label='cr')
    # # ax.scatter(*wl.T, c='g', marker='x', label='wl')
    # # ax.scatter(*wr.T, c='r', marker='x', label='wr')
    # ax.set_xlabel('x (mm)')
    # ax.set_ylabel('y (mm)')
    # ax.set_zlabel('z (mm)')
    # plt.legend()
    # plt.tight_layout()
    # plt.show()
    # plt.close()


if __name__ == '__main__':
    p = configargparse.ArgParser(default_config_files=['./config/config.ini'],
                                 ignore_unknown_config_file_keys=True)
    p.add('--config', is_config_file=True, help='Config file path')
    p.add('--recording', type=str, help='Input recording (json)')
    p.add('--output_dataframe', type=str,
          help='Output file for parsed recording')
    p.add('--n1', type=float, help='Refractive index n1')
    p.add('--n2', type=float, help='Refractive index n2')
    p.add('--eye_R', type=float, help='Eye R')
    p.add('--eye_K', type=float, help='Eye K')
    p.add('--eye_alpha', type=float, nargs=2, help='Eye alpha')
    p.add('--eye_beta', type=float, help='Eye beta')
    p.add('--cam_center', type=float, nargs=2,
          help='Camera center from calibration (in px)')
    p.add('--cam_focal', type=float, help='Camera focal length')
    p.add('--cam_pp', type=float, help='Camera pixel pitch')
    p.add('--cam_phi', type=float, help='Camera rotation around x-axis')
    p.add('--cam_theta', type=float, help='Camera rotation around y-axis')
    p.add('--cam_kappa', type=float, help='Camera rotation around y-axis')
    p.add('--screen_res', type=int, nargs=2, help='Screen resolution')
    p.add('--screen_pp', type=float, help='Screen pixel pitch')
    p.add('--screen_off', type=float, nargs=3,
          help='Screen offset from wcs center (lower border)')
    p.add('--cam_res', type=int, nargs=2, help='Camera resolution')
    p.add('--light_l', type=float, nargs=3, help='Position of left IR light')
    p.add('--light_r', type=float, nargs=3, help='Position of right IR light')
    options = p.parse_args()
    main(options)
