#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


cam1_matrix = np.array([
    [3.28096059e+03, 0.00000000e+00, 9.69093680e+02],
    [0.00000000e+00, 3.28086485e+03, 8.28389419e+02],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
cam1_dist = np.array([-1.18342721e-01,
                      6.40528615e-02,
                      6.70611000e-05,
                      -1.27527496e-03])
cam2_matrix = np.array([
    [3.36073182e+03, 0.00000000e+00, 1.19890016e+03],
    [0.00000000e+00, 3.35485021e+03, 7.90126077e+02],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
cam2_dist = np.array([-1.34503278e-01,
                      1.97805440e-01,
                      1.25122839e-04,
                      4.13394379e-03])


def stereo_calibrate(args, square_size=24.7):
    term_crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((1, 6*9, 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2) * square_size

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    # image list for plotting (with chessboard corners)
    images = []

    for image in [args.cam1, args.cam2]:
        img = cv2.imread(image[0])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # this is specific to the image - we have two chessboards in the image,
        # and for this we need to use the left one
        if image == args.cam2:
            gray[:, 1000:] = 0

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)
            cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), term_crit)
            imgpoints.append([corners])
            cv2.drawChessboardCorners(img, (9, 6), corners, ret)
            images.append(img)

    objpoints = np.array(objpoints)

    if args.plot:
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(images[0])
        ax[1].imshow(images[1])
        plt.show()

    _, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
        objpoints[0], imgpoints[0], imgpoints[1],
        cam1_matrix, cam1_dist,
        cam2_matrix, cam2_dist,
        gray.shape[::-1], flags=cv2.CALIB_FIX_INTRINSIC, criteria=term_crit)

    return R, T, E, F


def construct_homogeneous_transform(R, T, a=1):
    return np.vstack([np.hstack([R, T]), [0, 0, 0, a]])


def point_to_homogeneous(p):
    if len(p) == 2:
        return np.hstack([p, [0, 1]])
    if len(p) == 3:
        return np.hstack([p, 1])


def point_from_homogeneous(p):
    return p[0:-1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cam1', type=str, nargs=1, required=True,
                        help='Calibration image of camera 1.')
    parser.add_argument('--cam2', type=str, nargs=1, required=True,
                        help='Calibration image of camera 2.')
    parser.add_argument('--plot', '-p', action='store_true',
                        help='Show calibration plots.')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print translation and rotation matrices.')
    args = parser.parse_args()
    R, T, E, F = stereo_calibrate(args)

    if args.verbose:
        print('\n\nSTEREO CALIBRATION RESULTS\n')
        print('Rotation matrix:\n{}'.format(R))
        print('Translation matrix:\n{}'.format(T))
        print('Essential matrix:\n{}'.format(E))
        print('Fundamental matrix:\n{}'.format(F))

    img = cv2.imread(args.cam2[0])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray[:, :1000] = 0
    term_crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((1, 6*9, 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2) * 36.5
    _, corners = cv2.findChessboardCorners(gray, (9, 6), None)

    _, rvec, tvec = cv2.solvePnP(objp, corners, cam2_matrix, cam2_dist)

    if args.plot:
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(img)
        ax[1].imshow(gray, cmap='gray')
        plt.show()

    rmat = cv2.Rodrigues(rvec)[0]

    if args.verbose:
        print('\n\nRELATIVE CAMERA POSITIONS\n')
        print('Rotation matrix:\n{}'.format(rmat))
        print('Translation matrix:\n{}'.format(tvec))

    T1 = construct_homogeneous_transform(R, T)
    T2 = construct_homogeneous_transform(rmat, tvec)

    T1T2 = np.linalg.inv(T1).dot(T2)
    C1_h = np.array([0, 0, 0, 1])
    C2_h = np.linalg.inv(T1).dot(C1_h)
    C1 = point_from_homogeneous(C1_h)
    C2 = point_from_homogeneous(C2_h)
    objp_3d = []
    for op in objp[0]:
        obj_pos = T1T2.dot(point_to_homogeneous(op))
        objp_3d.append(point_from_homogeneous(obj_pos))
    objp_3d = np.array(objp_3d)

    vec1 = objp_3d[0] + objp_3d[8]
    vec2 = objp_3d[0] + objp_3d[-9]
    norm_vec = np.cross(vec1, vec2) / np.linalg.norm(np.cross(vec1, vec2))
    print('\n\nCenter point of screen:\n{}'.format(objp_3d[31]))
    print('\nScreen normal vector:\n{}'.format(norm_vec))

    if args.plot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(*objp_3d.T)
        ax.scatter(*(objp_3d[31] + 50 * norm_vec).T)
        ax.scatter(*(objp_3d[31] + 100 * norm_vec).T)
        ax.scatter(*C1)
        ax.scatter(*C2)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.show()
