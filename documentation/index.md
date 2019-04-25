# PyOSB documentation

PyOSB is a Python module that calculates a gaze ray from EyeOSB recordings (in JSON format).
The module consists of a [parser](/pyosb/fileio/eyeosbparser.py) that converts the JSON recordings to a [pandas DataFrame](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html), a [preprocessor](/pyosb/preprocess/preprocess.py) that rescales and rotates the recording data, and a [mapper](/pyosb/gaze/mapper.py) that does the actual gaze ray estimation.
Additionally, the module contains an unfinished [rabbitMQ receiver class](/pyosb/rabbitmq/receiver.py) that was originally intended to be able to receive the recording data live from EyeOSB.

The documentation is split into the following sections:

- [Parser](parser.md)
- [Preprocessor](preprocessor.md)
- [Mapper](mapper.md)
