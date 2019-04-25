# Mapper

The gaze ray calculation presented here is an implementation of the model-based gaze estimation technique described in [E.D. Guestrin's PhD thesis](https://tspace.library.utoronto.ca/handle/1807/24349) for a system of one camera and two light sources.
The functions in [mapper.py](/pyosb/gaze/mapper.py) contain a reference to numbered equations or sections in this thesis where applicable.
All calculations are based on a spherical cornea model.
Variable names follow the same conventions as the thesis and can be understood from the following ray-tracing diagram:
![eye model](images/eye_diagram.png)

The class consists of two main user functions, namely `GazeMapper.calc_gaze()` and `GazeMapper.calibrate()`, which are used for calculating the gaze ray and for user calibration, respectively.

## Initialization

The gaze mapper class is initialized using the `dict` returned from `Preprocessor.get_wcs_data()` and configuration values containing the camera intrinsics and the system setup (see [config.ini](/config/config.ini)).

## Gaze estimation
