# Preprocessor

The `Preprocessor` class does a few things:

1. Convert the pupil and reflex positions to the correct pixel values because EyeOSB stores them relative to the eye patch.
2. If the recording already contains gaze points (useful for comparisons), rescale them from $`\left[0, 1 \right]`$ to mm and recenter their origin to coincide with the center of the screen. Afterwards, rotate them using the extrinsic calibration (see [Camera calibration](camera_calibration.md)) so that the WCS origin sits at the center of the camera image plane (camera sensor).
3. Recenter reflex and pupil coordinates so that they coincide with the calibrated center of the camera image sensor (WCS origin) and rescale them from pixels to mm.
4. Calculate the positions of the light sources (measured by hand as offsets from the upper edge of the screen) in WCS through translation and rotations.
5. Add third axis to 2D image points ($`z = 0`$ on camera plane).
6. Drop all frames which don't have pupil and reflex positions.

## Usage

The preprocessor is initialized with the pandas DataFrame returned from `EyeInfoParser.get_dataframe()` and the config options containing the camera intrinsics, the screen information (resolution, center, normal), and the position of the light sources (see [config.ini](/config/config.ini)).

Afterwards, the preprocessed data can be returned as a `dict` using `Preprocessor.get_wcs_data()`, which the `GazeMapper` then uses for gaze estimation.

