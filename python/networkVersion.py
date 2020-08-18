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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import yaml
import htm2d.objectSpace
import htm2d.agent
from htm2d.agent import Direction

## ADD THIS TO location_network_creatinon for DEBUGGING ONLY
#sys.path.append('/media/D/Data/HTM/HTMpandaVis/pandaBaker')# DELETE AFTER DEBUGGING!!!
#from pandaNetwork import Network

import numpy as np
from htm.bindings.encoders import ScalarEncoder, ScalarEncoderParameters

from htm.advanced.frameworks.location.location_network_creation import L246aNetwork
from htm.advanced.support.register_regions import registerAllAdvancedRegions

logging.basicConfig(level=logging.ERROR)

_EXEC_DIR = os.path.dirname(os.path.abspath(__file__))
# go one folder up and then into the objects folder
_OBJECTS_DIR = os.path.join(_EXEC_DIR, os.path.pardir, "objects")

class Experiment:

    def __init__(self, mapSize=20):
        # create object space and the agent
        self.objSpace = htm2d.objectSpace.TwoDimensionalObjectSpace(mapSize, mapSize) # rectangle map
        self.agent = htm2d.agent.Agent()
        self.agent.set_objectSpace(self.objSpace, 0, 0)


    def loadObject(self, objectFilename):  # loads object into object space

        # load object from yml file
        with open(os.path.join(_OBJECTS_DIR, objectFilename), "r") as stream:
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
    def CreateSensationStream(self, n, w, type = "all" ): # this will create stream of pairs [location, sensation]

        stream = []
        # Create scalar encoder to encode features
        p = ScalarEncoderParameters()
        p.size = n
        p.activeBits = w
        p.minimum = 0
        p.maximum = 2
        encoder = ScalarEncoder(p)

        if type == "all": # agent will traverse every position in object space
            for x in range(self.objSpace.width):
                for y in range(self.objSpace.height):
                    self.agent.move(x, y)
                    feature = 1 if self.agent.get_feature(Direction.UP) == "X" else 0
                    stream.append(([x,y], list(encoder.encode(feature).sparse)))

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
        self.network = L246aNetwork(numColumns=1,
                                    L2Params=L2Params,
                                    L4Params=L4Params,
                                    L6aParams=L6aParams,
                                    repeat=self.numLearningPoints,
                                    logCalls=self.debug)


        sampleSize = L4Params["sampleSize"]
        columnCount = L4Params["columnCount"]

        # Make sure w is odd per encoder requirement
        sampleSize = sampleSize if sampleSize % 2 != 0 else sampleSize + 1

        # Load objects
        self.loadObject("a.yml")
        self.object1 = self.CreateSensationStream(type="all", w=sampleSize, n=columnCount)
        self.loadObject("b.yml")
        self.object2 = self.CreateSensationStream(type="all", w=sampleSize, n=columnCount)

        # Number of iterations must match the number of objects. This will allow us
        # to execute one iteration per object and use the "iteration" parameter as
        # the object index
        #assert params["iterations"] == len(self.objects)

        streamForAllColumns = {"object1" : [self.object1], "object2" : [self.object2]} # we are feeding now just for one column

        # Learn objects
        self.network.learn(streamForAllColumns)
        global objs
        objs = self.network.learnedObjects


    def infer(self, iteration):
        """
        For each iteration try to infer the object represented by the 'iteration'
        parameter returning Whether or not the object was unambiguously classified.
        :param params: Specific parameters for this iteration. See 'experiments.cfg'
                                     for list of parameters
        :param repetition: Current repetition
        :param iteration: Use the iteration to select the object to infer
        :return: Whether or not the object was classified
        """
        objname, sensations = list(self.objects.items())[iteration]

        # Select sensations to infer
        np.random.shuffle(sensations[0])
        sensations = [sensations[0][:self.numOfSensations]]

        self.network.sendReset()

        # Collect all statistics for every inference.
        # See L246aNetwork._updateInferenceStats
        stats = defaultdict(list)
        self.network.infer(sensations=sensations, stats=stats, objname=objname)
        stats.update({"name": objname})
        return stats




if __name__ == "__main__":
    registerAllAdvancedRegions()

    f = open("parameters.cfg", "r").read()
    parameters = eval(f)

    experiment = Experiment()

    experiment.learn(parameters, 0)

    #print(experiment.infer(0))


