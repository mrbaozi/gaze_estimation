#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import numpy as np
from pprint import pprint

with open('./recordings/evaluation_grid_9x9.json') as f:
    data = json.load(f)

eyecoords = []
pupilpos = []
reflexpos = []

for frame in data:
    if ('pupilpos' in frame['left_eye']) \
            and ('pupilpos' in frame['right_eye']) \
            and ('reflexpos' in frame['left_eye']) \
            and ('reflexpos' in frame['right_eye']):
        eyecoords.append([frame['left_eye']['eyecoords'], frame['right_eye']['eyecoords']])
        pupilpos.append([frame['left_eye']['pupilpos'], frame['right_eye']['pupilpos']])
        reflexpos.append([frame['left_eye']['reflexpos'], frame['right_eye']['reflexpos']])

ecl, ecr = [], []
ppl, ppr = [], []
rxl, rxr = [], []
for coord, ppos, rpos in zip(eyecoords, pupilpos, reflexpos):
    ecl.append([float(coord[0]['x']), float(coord[0]['y'])])
    ecr.append([float(coord[1]['x']), float(coord[1]['y'])])
    ppl.append([float(ppos[0]['x']), float(ppos[0]['y'])])
    ppr.append([float(ppos[1]['x']), float(ppos[1]['y'])])
    rxl.append([rpos[0][0]['x'], rpos[0][0]['y'], rpos[0][1]['x'], rpos[0][1]['y']])
    rxr.append([rpos[1][0]['x'], rpos[1][0]['y'], rpos[1][1]['x'], rpos[1][1]['y']])

ecl = np.array(ecl)
ecr = np.array(ecr)
ppl = np.array(ppl)
ppr = np.array(ppr)
rxl = np.array(rxl)
rxr = np.array(rxr)

ppl += ecl
ppr += ecr
for i in range(len(ecl)):
    rxl[i][0] += ecl[i][0]
    rxl[i][1] += ecl[i][1]
    rxl[i][2] += ecl[i][0]
    rxl[i][3] += ecl[i][1]
    rxr[i][0] += ecr[i][0]
    rxr[i][1] += ecr[i][1]
    rxr[i][2] += ecr[i][0]
    rxr[i][3] += ecr[i][1]

np.savetxt('./data/pupilpos_lefteye.txt', ppl)
np.savetxt('./data/pupilpos_righteye.txt', ppr)
np.savetxt('./data/reflexpos_lefteye.txt', rxl)
np.savetxt('./data/reflexpos_righteye.txt', rxr)
