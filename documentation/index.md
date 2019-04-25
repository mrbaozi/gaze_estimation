# PyOSB documentation

PyOSB is a Python module that calculates a gaze ray from EyeOSB recordings (in JSON format).
The module consists of a [parser](/pyosb/fileio/eyeosbparser.py) that converts the JSON recordings to a [pandas DataFrame](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html), a [preprocessor](/pyosb/preprocess/preprocess.py) that rescales and rotates the recording data, and a [mapper](/pyosb/gaze/mapper.py) that does the actual gaze ray estimation.
Additionally, the module contains an unfinished [rabbitMQ receiver class](/pyosb/rabbitmq/receiver.py) that was originally intended to be able to receive the recording data live from EyeOSB.

The documentation is split into the following sections:

- [Camera and system calibration](camera_calibration.md)
- [Parser](parser.md)
- [Preprocessor](preprocessor.md)
- [Mapper](mapper.md)

## Possible improvements

The implemented code provides a basic framework for doing model-based gaze estimation.
However, there is still a lot of potential for improvement.
Here I will list some things that come to mind:

- Implement an ellipsoidal corneal model as described in section 2.2.3.2 of [E.D. Guestrin's PhD thesis](https://tspace.library.utoronto.ca/handle/1807/24349). Currently, the calculations are based on a spherical model.
- Figure out how to properly calibrate the radius of corneal curvature $`R`$ and replace the calibration of the optical and visual axis offset angles $`\alpha`$ and $`\beta`$ with a single-point calibration.
- More robust calibration procedure in general, ideally online (while the eyetracker is running) with incremental refinements. This is a big task.
- Develop an easier way to calibrate the system. The current approach is... not good (see [Camera and system calibration](camera_calibration.md)).
- Better system setup in general with fixed cameras, fixed lights and proper measurements.
- Implement the gaze calculation for setups with different numbers of cameras and light sources. The equations for this are mostly the same, so this might actually be easier than it sounds. Stereo cameras might also eliminate the problem of calibrating $`R`$.
- Combine the method with multi-monitor eyetracking. In principle, this could be accomplished by intersecting the gaze with a larger scene.
- Flexible way to describe the scene with which to intersect the gaze ray, ideally using some kind of 3D model. Currently, the point of gaze is only calculated from the intersection with a plane.
- Sensitivity analysis of how the error in the point of gaze calculation changes with respect to changes in the system geometry (position of light sources, cameras, ...). Additionally, it would also be nice to know how exactly the values of $`R`$ and $`K`$ factor into the final point of gaze.
