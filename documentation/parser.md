# Parser

The parser implemented in [eyeosbparser.py](/pyosb/fileio/eyeosbparser.py) converts EyeOSB JSON recordings to pandas DataFrames and can optionally save it to an output file.
Most of what the parser does is trivial, the only noteworthy exception being that it modifies the JSON data so that pandas can interpret it correctly.
This preprocessing step is implemented and documented in `EyeInfoParser.preprocess()`.

## Usage

The `EyeInfoParser` is initialized with a path to the EyeOSB JSON recording.
The DataFrame can then simply be returned with a call to `EyeInfoParser.get_dataframe()`.
