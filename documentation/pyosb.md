# PyOSB

PyOSB is a Python module that calculates a gaze ray from EyeOSB recordings (in JSON format).
The module consists of a [parser](/pyosb/fileio/eyeosbparser.py) that converts the JSON recordings to a [pandas DataFrame](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html), a [preprocessor](/pyosb/preprocess/preprocess.py) that rescales and rotates the recording data, and a [mapper](/pyosb/gaze/mapper.py) that does the actual gaze ray estimation.
Additionally, the module contains an unfinished [rabbitMQ receiver class](/pyosb/rabbitmq/receiver.py) that was originally intended to be able to receive the recording data live from EyeOSB.

## Parser

The parser implemented in [eyeosbparser.py](/pyosb/fileio/eyeosbparser.py) converts EyeOSB JSON recordings to pandas DataFrames and can optionally save it to an output file.
Most of what the parser does is trivial, the only noteworthy exception being that it modifies the JSON data so that pandas can interpret it correctly.
This preprocessing step is implemented and documented in `EyeInfoParser.preprocess()`.

## Preprocessor

## Mapper

The gaze ray calculation presented here is an implementation of the model-based gaze estimation technique described in [E.D. Guestrin's PhD thesis](https://tspace.library.utoronto.ca/handle/1807/24349) for a system of one camera and two light sources.
The functions in [mapper.py](/pyosb/gaze/mapper.py) contain a reference to numbered equations or sections in this thesis where applicable.
All calculations are based on a spherical cornea model.
Variable names follow the same conventions as the thesis and can be understood from the following ray-tracing diagram:
![eye model](images/eye_diagram.png)


