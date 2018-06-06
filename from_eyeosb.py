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
    pp.rescale_to_screen('gaze_target')
    pp.fix_reflex_positions()
    pp.fliplr()
    pp.flipud()

    # final dataframe
    df = pp.get_dataframe()
    print(df.keys())
    df.plot.scatter(x='left_eye.pupilpos.x', y='left_eye.pupilpos.y')
    # df.plot.scatter(x='gaze_target.x', y='gaze_target.y')
    plt.show()

    # mapper = GazeMapper()


if __name__ == '__main__':
    p = configargparse.ArgParser(default_config_files=['./config/config.ini'],
                                 ignore_unknown_config_file_keys=True)
    p.add('--config', is_config_file=True, help='Config file path')
    p.add('--recording', type=str, help='Input recording (json)')
    p.add('--output_dataframe', type=str,
          help='Output file for parsed recording')
    p.add('--screen_x', type=int, help='Screen x resolution')
    p.add('--screen_y', type=int, help='Screen y resolution')
    p.add('--cam_x', type=int, help='Screen x resolution')
    p.add('--cam_y', type=int, help='Screen y resolution')
    options = p.parse_args()
    main(options)
