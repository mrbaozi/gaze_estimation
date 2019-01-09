#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


cam_matrix = np.array([
    [3.36073182e+03, 0.00000000e+00, 1.19890016e+03],
    [0.00000000e+00, 3.35485021e+03, 7.90126077e+02],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
cam_dist = np.array([-1.34503278e-01,
                     1.97805440e-01,
                     1.25122839e-04,
                     4.13394379e-03])


if __name__ == '__main__':
    img = cv2.imread('./18_31_02_121.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    term_crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((1, 36*21, 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:36, 0:21].T.reshape(-1, 2) * 13.22
    ret, corners = cv2.findChessboardCorners(gray, (36, 21), None)

    print(ret)

    images = []
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    # If found, add object points, image points (after refining them)
    if ret:
        objpoints.append(objp)
        cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), term_crit)
        imgpoints.append([corners])
        cv2.drawChessboardCorners(img, (9, 6), corners, ret)
        images.append(img)

    objpoints = np.array(objpoints)

    # fig, ax = plt.subplots(1, 1)
    # ax.imshow(images[0])
    # plt.show()

#     _, rvec, tvec = cv2.solvePnP(objp, corners, cam_matrix, cam_dist)

#     fig, ax = plt.subplots(1, 2)
#     ax[0].imshow(img)
#     ax[1].imshow(gray, cmap='gray')
#     plt.show()
