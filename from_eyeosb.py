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
    data = pp.get_wcs_data()

    df = pp.get_dataframe()
    for key in df.keys():
        print("{0: <30} {1}".format(key, df[key][0]))

    fig, ax = plt.subplots(2, 1)
    ax[0].scatter(df['left_eye.pupilpos.x'], df['left_eye.pupilpos.y'])
    ax[0].scatter(df['left_eye.reflexpos.left.x'],
                  df['left_eye.reflexpos.left.y'], marker='.')
    ax[0].scatter(df['left_eye.reflexpos.right.x'],
                  df['left_eye.reflexpos.right.y'], marker='.')
    ax[0].scatter(df['right_eye.pupilpos.x'], df['right_eye.pupilpos.y'])
    ax[0].scatter(df['right_eye.reflexpos.left.x'],
                  df['right_eye.reflexpos.left.y'], marker='.')
    ax[0].scatter(df['right_eye.reflexpos.right.x'],
                  df['right_eye.reflexpos.right.y'], marker='.')
    ax[1].scatter(df['gaze_target.x'], df['gaze_target.y'])
    ax[1].scatter(df['gaze_point.x'], df['gaze_point.y'], marker='x')
    ax[1].scatter(df['left_eye.gazepos.x'],
                  df['left_eye.gazepos.y'], marker='.')
    ax[1].scatter(df['right_eye.gazepos.x'],
                  df['right_eye.gazepos.y'], marker='.')
    ax[1].set_xlim([-options.screen_res[0] * options.screen_pp / 2,
                    options.screen_res[0] * options.screen_pp / 2])
    ax[1].set_ylim([options.screen_res[1] * options.screen_pp, 0])
    plt.show()

    mapper = GazeMapper(options, data)


if __name__ == '__main__':
    p = configargparse.ArgParser(default_config_files=['./config/config.ini'],
                                 ignore_unknown_config_file_keys=True)
    p.add('--config', is_config_file=True, help='Config file path')
    p.add('--recording', type=str, help='Input recording (json)')
    p.add('--output_dataframe', type=str,
          help='Output file for parsed recording')
    p.add('--n1', type=float, help='Refractive index n1')
    p.add('--n2', type=float, help='Refractive index n2')
    p.add('--R', type=float, help='Eye R')
    p.add('--K', type=float, help='Eye K')
    p.add('--alpha', type=float, help='Eye alpha')
    p.add('--beta', type=float, help='Eye beta')
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
