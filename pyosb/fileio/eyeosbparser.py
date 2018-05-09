#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import pandas as pd
from pandas.io.json import json_normalize


class EyeInfoParser():
    def __init__(self, args=None):
        if args == None:
            print("No arguments given to EyeInfoParser constructor. " +
                  "Either pass args to constructor or specify them explicitly.")
        else:
            self.experimentsFolder = args.experimentsfolder
            self.outputFile = args.parsedf
            self.jsonFiles = self.get_filenames()
            self.jsonData = self.load_json()
            self.dataFrames = self.json_to_pandas()
            self.dataFrame = pd.concat(self.dataFrames)

    def get_filenames(self, folder=None):
        if folder == None:
            folder = self.experimentsFolder

        files = []
        for dirpath, dirnames, filenames in os.walk(folder):
            for filename in [f for f in filenames if f.endswith(".json")]:
                files.append(os.path.join(dirpath, filename))

        return files

    def load_json(self, jsonfiles=None):
        if jsonfiles == None:
            jsonfiles = self.jsonFiles

        jsondata = []
        for jsonfile in jsonfiles:
            with open(jsonfile) as data:
                jsondata.append(json.loads(self.preprocess(data)))

        return jsondata

    def json_to_pandas(self, jsonfiles=None, jsondata=None):
        if jsonfiles == None:
            jsonfiles = self.jsonFiles
        if jsondata == None:
            jsondata = self.jsonData

        dataframes = []
        for experiment, data in zip(jsonfiles, jsondata):
            df = json_normalize(data)
            df['experiment'] = os.path.splitext(os.path.basename(experiment))[0]
            dataframes.append(df)

        return dataframes

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

        self.dataFrame.to_pickle("./" + filename)
