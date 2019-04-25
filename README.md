# PyOSB

This module implements a model-based gaze estimation technique based on [E.D. Guestrin's PhD thesis](https://tspace.library.utoronto.ca/handle/1807/24349).

## Requirements

- Python >= 3.6
- numpy
- scipy
- matplotlib
- pandas
- configargparse
- tqdm

## Installation

The easiest way to install PyOSB is using pip and the included `setup.py`:
```
cd /path/to/pyosb
pip install .
```
While working on PyOSB, it is convenient to `pip install` it with the `--editable` option (preferrably in a [virtual environment](https://virtualenv.pypa.io/en/latest/#)):
```
cd /path/to/pyosb
pip install -e .
```

## Usage

1. Set the path to the EyeOSB JSON recording in `[eyeosb_parser]` in `config.ini`.
2. Create an `EyeInfoParser` object with this config and call `get_dataframe()`.
3. Preprocess this DataFrame by creating a `Preprocessor` object and calling `get_wcs_data()`.
4. Pass this data to the `GazeMapper` and call `calc_gaze()` to get the gaze estimation.

A basic usage example is given in [run.py](/run.py).

## Documentation

The documentation can be found [here](/documentation/index.md).
