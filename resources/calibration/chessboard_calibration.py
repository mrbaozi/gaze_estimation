#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys
import argparse
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt


def calibrate(folder, square_size=25, save_output=False):
    term_crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2) * square_size

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    for fname in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, fname))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)
            cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), term_crit)
            imgpoints.append(corners)

    objpoints = np.array(objpoints)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints,
                                                       gray.shape[::-1],
                                                       None, None,
                                                       criteria=term_crit)
    fovx, fovy, focalLength, principalPoint, aspectRatio = \
        cv2.calibrationMatrixValues(mtx, gray.shape[::-1], 5.2, 3.88)

    if save_output:
        with open(os.path.basename(os.path.normpath(folder))
                  + '_calibration.txt', 'w') as ofile:
            ofile.write("Average reprojection error: {}\n".format(ret))
            ofile.write("Camera intrinsic matrix:\n{}\n".format(mtx))
            ofile.write("Distortion coefficients:\n{}\n".format(dist))
            ofile.write("Per image rotation vectors:\n{}\n".format(rvecs))
            ofile.write("Per image translation vectors:\n{}\n"
                        .format(tvecs))
            ofile.write("Field of view in degress along "
                        "horizontal sensor axis:\n{}\n".format(fovx))
            ofile.write("Field of view in degress along "
                        "vertical sensor axis:\n{}\n".format(fovy))
            ofile.write("Focal length of lens in mm:\n{}\n"
                        .format(focalLength))
            ofile.write("Principal point in mm:\n{}\n"
                        .format(principalPoint))
            ofile.write("Pixel aspect ratio:\n{}".format(aspectRatio))

    return [ret, mtx, dist, rvecs, tvecs]


def main(args):
    calibrations = []
    for folder in args.folders:
        calibrations.append(calibrate(folder, 25, args.save))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folders', '-f', type=str, nargs='+', required=True,
                        help='List of folders with calibration images')
    parser.add_argument('--save', '-s', action='store_true',
                        help='Save calibrations to file')
    args = parser.parse_args()
    main(args)
