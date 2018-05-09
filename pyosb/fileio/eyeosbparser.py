#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import pandas as pd
from pandas.io.json import json_normalize


class EyeInfoParser(object):
    def __init__(self, args=None):
        if args == None:
            print("No arguments given to EyeInfoParser constructor. " +
                  "Either pass args to constructor or specify them explicitly.")
        else:
            self.recording = args.recording
            self.outputFile = args.output_dataframe
            self.jsonData = self.load_json()
            self.dataFrame = self.json_to_pandas()

    def load_json(self, jsonfile=None):
        if jsonfile == None:
            jsonfile = self.recording
        with open(jsonfile) as data:
            jsondata = json.loads(self.preprocess(data))
        return jsondata

    def json_to_pandas(self, jsondata=None):
        if jsondata == None:
            jsondata = self.jsonData
        df = json_normalize(jsondata)
        return df

    def preprocess(self, jsonfile):
        """
        This function loads the json recording as a string and performs regex
        matching on malformed json. Doing this here enables pandas' json loader
        to load eyetracker recordings properly. Should the json format of
        eyetracker recordings change in the future fix it in this function so
        no other changes to the parser have to be made.
        """
        s = jsonfile.read()

        # eyecoords need x, y identifiers
        rx = r'"eyecoords": "\[(\d+\.?\d*), (\d+\.?\d*)\]"'
        rxfix = r'"eyecoords": {"x": \1, "y": \2}'
        s = re.sub(rx, rxfix, s)

        # reflex positions need unique identifiers
        rx = r'"reflexpos": \[\{"x": (\d+\.?\d*), "y": (\d+\.?\d*)\}, \{"x": (\d+\.?\d*), "y": (\d+\.?\d*)\}\]'
        rxfix = r'"reflexpos": {"left": {"x": \1, "y": \2}, "right": {"x": \3, "y": \4}}'
        s = re.sub(rx, rxfix, s)

        return s

    def df_to_file(self, filename=None):
        if filename == None:
            filename = self.outputFile
        self.dataFrame.to_pickle(filename)

    def get_dataframe(self):
        return self.dataFrame
