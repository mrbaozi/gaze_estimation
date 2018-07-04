#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


def main():
    targets = np.loadtxt('../data/targets.txt')
    ppos_left = np.loadtxt('../data/pupilpos_lefteye.txt')
    ppos_right = np.loadtxt('../data/pupilpos_righteye.txt')
    reflex_left = np.loadtxt('../data/reflexpos_lefteye.txt')
    reflex_right = np.loadtxt('../data/reflexpos_righteye.txt')

    print(reflex_left.T)

    # sanity check if targets are ok
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    ax.scatter(*targets.T, marker='x', s=5)
    plt.show()

    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].scatter(*ppos_left.T, marker='.', s=1)
    ax[0].scatter(reflex_left.T[0], reflex_left.T[1], marker='.', s=1)
    ax[0].scatter(reflex_left.T[2], reflex_left.T[3], marker='.', s=1)
    ax[1].scatter(*ppos_right.T, marker='.', s=1)
    ax[1].scatter(reflex_right.T[0], reflex_right.T[1], marker='.', s=1)
    ax[1].scatter(reflex_right.T[2], reflex_right.T[3], marker='.', s=1)
    plt.show()


if __name__ == '__main__':
    main()
