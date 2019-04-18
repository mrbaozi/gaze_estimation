# Camera calibration

## Mono calibration

## Stereo calibration

The gaze mapping calculation relies on a plane to intersect the gaze ray with, which is the screen in our case.
This plane is given by the screen center and the screen normal vector, labeled `screen_center` and `screen_norm` in `config/config.ini`, respectively.
