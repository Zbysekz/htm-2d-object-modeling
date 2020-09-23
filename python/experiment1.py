# ----------------------------------------------------------------------
# HTM Community Edition of NuPIC
# Copyright (C) 2017, Numenta, Inc.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.    If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

"""
    This script creates simple experiment to compute the object classification
    accuracy of L2-L4-L6 network using objects from YCB dataset and "Thing" sensor
"""
import glob
import json
import logging
import os
import random
from collections import defaultdict, OrderedDict
import copy

import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt

import yaml
import experimentFramework.objectSpace as objectSpace
import experimentFramework.agent as agent
from experimentFramework.agent import Direction

import numpy as np
from htm.bindings.encoders import ScalarEncoder, ScalarEncoderParameters

from l2l4l6Framework.l2_l4_l6_Network import L2_L4_L6_Network
from htm.advanced.support.register_regions import registerAllAdvancedRegions

from utilities import (
    isNotebook,
    plotEnvironment)

logging.basicConfig(level=logging.WARN)

_EXEC_DIR = os.path.dirname(os.path.abspath(__file__))
# go one folder up and then into the objects folder
_OBJECTS_DIR = os.path.join(_EXEC_DIR, os.path.pardir, "objects")

class Experiment:

    def __init__(self, objectSpaceSize):
        # create object space and the agent
        self.objSpace = objectSpace.TwoDimensionalObjectSpace(objectSpaceSize, objectSpaceSize) # rectangle object space - "map"
        self.agent = agent.Agent()
        self.agent.set_objectSpace(self.objSpace, 0, 0)
        self.learnedObjects = {}

        self.bakePandaData = False # bake or not data for PandaVis

        self.fig_environment = None


    def loadObject(self, objectFilename):  # loads object into object space

        # load object from yml file
        with open(os.path.join(_OBJECTS_DIR, objectFilename+".yml"), "r") as stream:
            try:
                self.objSpace.load_object(stream)
            except yaml.YAMLError as exc:
                print(exc)

    """
    :param n: The number of bits in the feature SDR. Usually L4 column count
    :type n: int
    :param w: Number of 'on' bits in the feature SDR. Usually L4 sample size
    :type w: int
    
    :return {'obj1' : [[[1,1,1],[101,205,523, ..., 1021]],...], ...}
    """
    def CreateSensationStream(self, sensorDirection, n, w, type = "all", sparsity = 0.1 ): # this will create stream of pairs [location, sensation]

        stream = []
        # Create scalar encoder to encode features
        p = ScalarEncoderParameters()
        p.size = n
        p.activeBits = w
        p.minimum = 0
        p.maximum = 1
        encoder = ScalarEncoder(p)

        row = list(range(0, self.objSpace.width))
        row_reverse = row.copy()
        row_reverse.reverse()

        # this simulates movement of the sensor like "snake" visiting each place in the space once
        # it is like: ------->|
        #             |<------ˇ
        #             ˇ------->
        # benefit is, that movement is continuous, with step size always 1
        x = np.concatenate([row if i % 2 == 0 else row_reverse for i in range(self.objSpace.height)])
        y = np.concatenate([[i] * self.objSpace.width for i in range(0, self.objSpace.height)])

        if type == "random_sample": # randomly pick n choices from the all possibilities
            ni = (int)(len(x) * sparsity)  # for 2 random indices
            index = np.random.choice(x.shape[0], ni, replace=False)

            x = x[index]
            y = y[index]
        else: #"all" # agent will traverse every position in object space
            pass

        for i in range(len(x)):
            self.agent.move(x[i], y[i])
            f = self.agent.get_feature(sensorDirection)
            feature = (('X', 'Y', 'Z').index(f)+1) if f is not None else 0
            stream.append(([x[i], y[i]], list(encoder.encode(feature).sparse)))

        return stream

    def learn(self, params, repetition):
        """
        Take the steps necessary to reset the experiment before each repetition:
            - Make sure random seed is different for each repetition
            - Create the L2-L4-L6a network
            - Load objects used by the experiment
            - Learn all objects used by the experiment
        """
        print(params["name"], ":", repetition)
        self.debug = params.get("debug", False)
        self.numLearningPoints = params["num_learning_points"]
        self.numOfSensations = params["num_sensations"]

        L2Params = params["l2_params"]
        L4Params = params["l4_params"]
        L6aParams = params["l6a_params"]

        self.sdrSize = L2Params["sdrSize"]

        # Make sure random seed is different for each repetition
        seed = params.get("seed", 42)
        np.random.seed(seed + repetition)
        random.seed(seed + repetition)
        L2Params["seed"] = seed + repetition
        L4Params["seed"] = seed + repetition
        L6aParams["seed"] = seed + repetition

        # Configure L6a params
        numModules = L6aParams["moduleCount"]
        L6aParams["scale"] = [params["scale"]] * numModules
        angle = params["angle"] // numModules
        orientation = list(range(angle // 2, angle * numModules, angle))
        L6aParams["orientation"] = np.radians(orientation).tolist()
        L6aParams["cellsPerAxis"] = params["cells_per_axis"]

        # Create single column L2-L4-L6a network
        self.network = L2_L4_L6_Network(numColumns=4,
                                    L2Params=L2Params,
                                    L4Params=L4Params,
                                    L6aParams=L6aParams,
                                    repeat=self.numLearningPoints,
                                    logCalls=self.debug)

        # data for dash plots
        self.network.network.updateDataStreams = self.updateDataStreams

        sampleSize = L4Params["sampleSize"]
        columnCount = L4Params["columnCount"]

        # Make sure w is odd per encoder requirement
        sampleSize = sampleSize if sampleSize % 2 != 0 else sampleSize + 1

        # Load objects
        learnedObjectNames = ["a", "b", "boat", "cup", "palmpilot"] #["a", "b", "boat", "cup", "palmpilot"]#["simple1", "simple2", "simple3"]


        streamForAllColumns = {}
        self.learnedObjects = {}
        for obj in learnedObjectNames:
            self.loadObject(obj) # loads object into object space
            self.learnedObjects[obj] = {}
            self.learnedObjects[obj][0] = self.CreateSensationStream(sensorDirection = Direction.UP,type="random_sample",
                                                        w=sampleSize, n=columnCount)
            self.learnedObjects[obj][1] = self.CreateSensationStream(sensorDirection=Direction.DOWN, type="random_sample",
                                                         w=sampleSize, n=columnCount)
            self.learnedObjects[obj][2] = self.CreateSensationStream(sensorDirection=Direction.LEFT, type="random_sample",
                                                         w=sampleSize, n=columnCount)
            self.learnedObjects[obj][3] = self.CreateSensationStream(sensorDirection=Direction.RIGHT, type="random_sample",
                                                         w=sampleSize, n=columnCount)

            streamForAllColumns[obj] = [self.learnedObjects[obj][0], self.learnedObjects[obj][1],
                                            self.learnedObjects[obj][2], self.learnedObjects[obj][3]] # we are feeding now just for one column

            self.plotStream(self.learnedObjects[obj][0])

        # Learn objects
        self.network.learn(streamForAllColumns)

    def plotStream(self, stream):

        for s in stream:
            # Plotting and visualising environment-------------------------------------------

            # Plotting and visualising environment-------------------------------------------
            if (
                    self.fig_environment == None or isNotebook()
            ):  # create figure only if it doesn't exist yet or we are in interactive console
                self.fig_environment, _ = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
            else:
                self.fig_environment.axes[0].clear()

            plotEnvironment(self.fig_environment.axes[0], "Environment", self.objSpace, s[0][0])
            self.fig_environment.canvas.draw()

            plt.show(block=False)
            plt.pause(0.001)  # delay is needed for proper redraw

    def updateDataStreams(self):

        # for first column
        col = 0
        self.network.network.UpdateDataStream("L2ActiveCellCnt", len(self.network.getL2Representations()[col]))
        self.network.network.UpdateDataStream("L4PredictedCellCnt", len(self.network.getL4PredictedCells()[col]))
        self.network.network.UpdateDataStream("L4ActiveCellCnt", len(self.network.getL4Representations()[col]))
        self.network.network.UpdateDataStream("L6ActiveCellCnt", len(self.network.getL6aRepresentations()[col]))

    def infer(self, objectName):
        """
        For each iteration try to infer the object.

        :param objectName: Object name to infer
        :return: stats of inferrence
        """

        sensations = copy.deepcopy(self.learnedObjects[objectName])

        #rng = np.random.default_rng()
        indexes = random.sample(range(0, len(sensations[0])), self.numOfSensations)

        sampledSensations = []
        for k, s in sensations.items():# for each sensor sensation
            s = np.array(s, dtype=object)
            sampledSensations.append([s[np.array(np.array(indexes))]])

        self.plotStream(sampledSensations[0]) # plot just first, same as the others

        self.network.sendReset()

        # Collect all statistics for every inference.
        # See L246aNetwork._updateInferenceStats
        stats = defaultdict(list)
        self.network.infer(sensations=sampledSensations, stats=stats, objname=objectName)
        stats.update({"name": objectName})

        return stats



if __name__ == "__main__":
    registerAllAdvancedRegions()

    with open("parameters.cfg", "r") as f:
        parameters = eval(f.read())

    experiment = Experiment(objectSpaceSize=20) # "map size" - adjust to fit objects into object space

    experiment.learn(parameters, 0)

    print("Learning done, begin inferring")

    for obj in ['cup','boat', 'palmpilot', 'a','b']:
        stats = experiment.infer(objectName=obj)

        if 1 in stats['Correct classification']:
            print("Correctly classified!!")

    printedStats = json.dumps(stats, indent=4)
    with open("stats.json", "w") as f:
        f.write(printedStats)


