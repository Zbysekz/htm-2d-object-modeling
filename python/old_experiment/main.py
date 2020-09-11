#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os.path
import yaml
import htm2d.environment
import htm2d.agent
from htm2d.agent import Direction
import numpy as np
import time
import matplotlib.pyplot as plt
import random

from utilities import (
    plotBinaryMap,
    isNotebook,
    plotEnvironment,
)  # auxiliary functions from utilities.py

from htm.bindings.algorithms import SpatialPooler, TemporalMemory
from htm.bindings.sdr import SDR, Metrics
from htm.encoders.rdse import RDSE, RDSE_Parameters
from htm.encoders.grid_cell_encoder import GridCellEncoder
from htm.algorithms.anomaly import Anomaly
from htm.advanced.algorithms.apical_tiebreak_temporal_memory import ApicalTiebreakPairMemory

PLOT_GRAPHS = True
PLOT_ENV = True
PANDA_VIS_BAKE_DATA = True # if we want to bake data for pandaVis tool (repo at https://github.com/htm-community/HTMpandaVis )

if PANDA_VIS_BAKE_DATA:
    from pandaBaker.pandaBaker import PandaBaker
    from pandaBaker.pandaBaker import cLayer, cInput, cDataStream

    BAKE_DATABASE_FILE_PATH = os.path.join(os.getcwd(), 'bakedDatabase', 'htmcore_detector.db')
    pandaBaker = PandaBaker(BAKE_DATABASE_FILE_PATH)

_EXEC_DIR = os.path.dirname(os.path.abspath(__file__))
# go one folder up and then into the objects folder
_OBJECTS_DIR = os.path.join(_EXEC_DIR, os.path.pardir, "objects")

OBJECT_FILENAME = "a.yml"  # what object to load

fig_layers = None
fig_graphs = None
fig_environment = None
fig_expect = None

class Experiment:

    def __init__(self):
        self.anomalyHistData = []
        self.iterationNo = 0

        self.enc_info = None
        self.sp_info = None
        self.tm_info = None

    def SystemSetup(self, verbose=True):

        if verbose:
            import pprint

            print("Parameters:")
            pprint.pprint(parameters, indent=4)
            print("")

        # create environment and the agent
        self.env = htm2d.environment.TwoDimensionalEnvironment(20, 20)
        self.agent = htm2d.agent.Agent()

        # load object from yml file
        with open(os.path.join(_OBJECTS_DIR, OBJECT_FILENAME), "r") as stream:
            try:
                self.env.load_object(stream)
            except yaml.YAMLError as exc:
                print(exc)

        # SENSOR LAYER --------------------------------------------------------------
        # setup sensor encoder
        sensorEncoderParams = RDSE_Parameters()
        sensorEncoderParams.category = True
        sensorEncoderParams.size = parameters["enc"]["size"]
        sensorEncoderParams.sparsity = parameters["enc"]["sparsity"]
        sensorEncoderParams.seed = parameters["enc"]["seed"]
        self.sensorEncoder = RDSE(sensorEncoderParams)

        # Create SpatialPooler

        self.sensorLayer_sp = SpatialPooler(
            inputDimensions=(self.sensorEncoder.size,),
            columnDimensions=(spParams["columnCount"],),
            potentialPct=spParams["potentialPct"],
            potentialRadius=self.sensorEncoder.size,
            globalInhibition=True,
            localAreaDensity=spParams["localAreaDensity"],
            synPermInactiveDec=spParams["synPermInactiveDec"],
            synPermActiveInc=spParams["synPermActiveInc"],
            synPermConnected=spParams["synPermConnected"],
            boostStrength=spParams["boostStrength"],
            wrapAround=True,
        )
        self.sp_info = Metrics(self.sensorLayer_sp.getColumnDimensions(), 999999999)

        # Create an SDR to represent active columns, This will be populated by the
        # compute method below. It must have the same dimensions as the Spatial Pooler.
        self.sensorLayer_SDR_columns = SDR(spParams["columnCount"])

        # LOCATION LAYER ------------------------------------------------------------
        # Grid cell modules

        self.gridCellEncoder = GridCellEncoder(
            size=locParams["cellCount"],
            sparsity=locParams["sparsity"],
            periods=locParams["periods"],
            seed=locParams["seed"],
        )

        self.locationlayer_SDR_cells = SDR(self.gridCellEncoder.dimensions)



        initParams = {
            "columnCount": spParams["columnCount"],
            "cellsPerColumn": tmParams["cellsPerColumn"],
            "basalInputSize": locParams["cellCount"],
            "activationThreshold" : tmParams["activationThreshold"],
            "reducedBasalThreshold" : 13,
            "initialPermanence" : tmParams["initialPerm"],
            "connectedPermanence" : spParams["synPermConnected"],
            "minThreshold" : tmParams["minThreshold"],
            "sampleSize" : 20,
            "permanenceIncrement" : tmParams["permanenceInc"],
            "permanenceDecrement" : tmParams["permanenceDec"],
            "basalPredictedSegmentDecrement" : 0.0,
            "apicalPredictedSegmentDecrement" : 0.0,
            "maxSynapsesPerSegment" : tmParams["maxSynapsesPerSegment"]
        }

        self.sensoryLayer_tm = ApicalTiebreakPairMemory(**initParams)


        # self.sensoryLayer_tm = TemporalMemory(
        #     columnDimensions=(spParams["columnCount"],),
        #     cellsPerColumn=tmParams["cellsPerColumn"],
        #     activationThreshold=tmParams["activationThreshold"],
        #     initialPermanence=tmParams["initialPerm"],
        #     connectedPermanence=spParams["synPermConnected"],
        #     minThreshold=tmParams["minThreshold"],
        #     maxNewSynapseCount=tmParams["newSynapseCount"],
        #     permanenceIncrement=tmParams["permanenceInc"],
        #     permanenceDecrement=tmParams["permanenceDec"],
        #     predictedSegmentDecrement=0.0,
        #     maxSegmentsPerCell=tmParams["maxSegmentsPerCell"],
        #     maxSynapsesPerSegment=tmParams["maxSynapsesPerSegment"],
        #     externalPredictiveInputs=locParams["cellCount"],
        # )
        self.tm_info = Metrics([self.sensoryLayer_tm.numberOfCells()], 999999999)

    def CellsToColumns(self, cells, cellsPerColumn, columnsCount):
        array  = []
        for cell in cells.sparse:
            col = int(cell/cellsPerColumn)
            if col not in array:#each column max once
                array += [col]

        columns = SDR(columnsCount)
        columns.sparse = array
        return columns

    def SystemCalculate(self, feature, learning):
        global fig_environment, fig_graphs
        # ENCODE DATA TO SDR--------------------------------------------------
        # Convert sensed feature to int
        self.sensedFeature = 1 if feature == "X" else 0
        self.sensorSDR = self.sensorEncoder.encode(self.sensedFeature)

        # ACTIVATE COLUMNS IN SENSORY LAYER ----------------------------------
        # Execute Spatial Pooling algorithm on Sensory Layer with sensorSDR as proximal input
        self.sensorLayer_sp.compute(self.sensorSDR, learning, self.sensorLayer_SDR_columns)

        # SIMULATE LOCATION LAYER --------------------------------------------
        # Execute Location Layer - it is just GC encoder
        self.gridCellEncoder.encode(self.agent.get_position(), self.locationlayer_SDR_cells)

        #
        # Execute Temporal memory algorithm over the Sensory Layer, with mix of
        # Location Layer activity and Sensory Layer activity as distal input
        externalDistalInput = self.locationlayer_SDR_cells

        tm_input = {
            "activeColumns": self.sensorLayer_SDR_columns.sparse,
            "basalInput": externalDistalInput.sparse,
            "basalGrowthCandidates": None,
            "learn": learning
        }
        self.sensoryLayer_tm.compute(**tm_input)

        #self.sensoryLayer_tm.activateCells(self.sensorLayer_SDR_columns, learning)

        # activateDendrites calculates active segments
        #self.sensoryLayer_tm.activateDendrites(learn=learning, externalPredictiveInputsActive=externalDistalInput,
                                         #externalPredictiveInputsWinners=externalDistalInput)
        # predictive cells are calculated directly from active segments
        self.predictiveCellsSDR = SDR(spParams["columnCount"]* tmParams["cellsPerColumn"])
        self.predictiveCellsSDR.sparse = self.sensoryLayer_tm.predictedCells

        if self.iterationNo!=0:
        # and calculate anomaly - compare how much of active columns had some predictive cells
            self.rawAnomaly = Anomaly.calculateRawAnomaly(self.sensorLayer_SDR_columns,
                                                 self.CellsToColumns(self.predictiveCellsSDR,parameters["sensoryLayer_tm"]["cellsPerColumn"],parameters["sensoryLayer_sp"]["columnCount"]))
        else:
            self.rawAnomaly = 0


        # PANDA VIS
        if PANDA_VIS_BAKE_DATA:
            # ------------------HTMpandaVis----------------------
            # fill up values
            pandaBaker.inputs["FeatureSensor"].stringValue = "Feature: {:.2f}".format(self.sensedFeature)
            pandaBaker.inputs["FeatureSensor"].bits = self.sensorSDR.sparse

            pandaBaker.inputs["LocationLayer"].stringValue = str(self.agent.get_position())
            pandaBaker.inputs["LocationLayer"].bits = self.locationlayer_SDR_cells.sparse

            pandaBaker.layers["SensoryLayer"].activeColumns = self.sensorLayer_SDR_columns.sparse
            pandaBaker.layers["SensoryLayer"].winnerCells =  self.sensoryLayer_tm.getWinnerCells()
            pandaBaker.layers["SensoryLayer"].predictiveCells = self.predictiveCellsSDR.sparse
            pandaBaker.layers["SensoryLayer"].activeCells = self.sensoryLayer_tm.getActiveCells()

            # customizable datastreams to be show on the DASH PLOTS
            pandaBaker.dataStreams["rawAnomaly"].value = self.rawAnomaly
            pandaBaker.dataStreams["numberOfWinnerCells"].value = len(self.sensoryLayer_tm.getWinnerCells())
            pandaBaker.dataStreams["numberOfPredictiveCells"].value = len(self.predictiveCellsSDR.sparse)
            pandaBaker.dataStreams["sensor_sparsity"].value = self.sensorSDR.getSparsity()*100
            pandaBaker.dataStreams["location_sparsity"].value = self.locationlayer_SDR_cells.getSparsity()*100

            pandaBaker.dataStreams["SensoryLayer_SP_overlap_metric"].value = self.sp_info.overlap.overlap
            pandaBaker.dataStreams["SensoryLayer_TM_overlap_metric"].value = self.sp_info.overlap.overlap
            pandaBaker.dataStreams["SensoryLayer_SP_activation_frequency"].value = self.sp_info.activationFrequency.mean()
            pandaBaker.dataStreams["SensoryLayer_TM_activation_frequency"].value = self.tm_info.activationFrequency.mean()
            pandaBaker.dataStreams["SensoryLayer_SP_entropy"].value = self.sp_info.activationFrequency.mean()
            pandaBaker.dataStreams["SensoryLayer_TM_entropy"].value = self.tm_info.activationFrequency.mean()

            pandaBaker.StoreIteration(self.iterationNo)

            # ------------------HTMpandaVis----------------------


        print("Position:" + str(self.agent.get_position()))
        print("Feature:" + str(self.sensedFeature))
        print("Anomaly score:" + str(self.rawAnomaly))
        self.anomalyHistData += [self.rawAnomaly]


        if PLOT_ENV:
            # Plotting and visualising environment-------------------------------------------
            if (
                    fig_environment == None or isNotebook()
            ):  # create figure only if it doesn't exist yet or we are in interactive console
                fig_environment, _ = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
            else:
                fig_environment.axes[0].clear()

            plotEnvironment(fig_environment.axes[0], "Environment", self.env, self.agent.get_position())
            fig_environment.canvas.draw()

            plt.show(block=False)
            plt.pause(0.001)  # delay is needed for proper redraw

        self.iterationNo += 1


        if PLOT_GRAPHS:
            # ---------------------------
            if (
                    fig_graphs == None or isNotebook()
            ):  # create figure only if it doesn't exist yet or we are in interactive console
                fig_graphs, _ = plt.subplots(nrows=1, ncols=1, figsize=(5, 2))
            else:
                fig_graphs.axes[0].clear()

            fig_graphs.axes[0].set_title("Anomaly score")
            fig_graphs.axes[0].plot(self.anomalyHistData)
            fig_graphs.canvas.draw()

            #if agent.get_position() != [3, 4]:  # HACK ALERT! Ignore at this pos (after reset)
            #    anomalyHistData += [sensoryLayer_tm.anomaly]




    def BuildPandaSystem(self):

        pandaBaker.inputs["FeatureSensor"] = cInput(self.sensorEncoder.size)

        pandaBaker.layers["SensoryLayer"] = cLayer(self.sensorLayer_sp, self.sensoryLayer_tm)
        pandaBaker.layers["SensoryLayer"].proximalInputs = ["FeatureSensor"]
        pandaBaker.layers["SensoryLayer"].distalInputs = ["LocationLayer"]


        pandaBaker.inputs["LocationLayer"] = cInput(self.gridCellEncoder.size) # for now, Location layer is just position encoder

        # data for dash plots
        streams = ["rawAnomaly", "numberOfWinnerCells", "numberOfPredictiveCells",
                   "sensor_sparsity", "location_sparsity", "SensoryLayer_SP_overlap_metric", "SensoryLayer_TM_overlap_metric",
                   "SensoryLayer_SP_activation_frequency", "SensoryLayer_TM_activation_frequency", "SensoryLayer_SP_entropy",
                   "SensoryLayer_TM_entropy"
                   ]

        pandaBaker.dataStreams = dict((name, cDataStream()) for name in streams)  # create dicts for more comfortable code
        # could be also written like: pandaBaker.dataStreams["myStreamName"] = cDataStream()

        pandaBaker.PrepareDatabase()


    def RunExperiment1(self):
        global fig_expect

        # put agent in the environment
        self.agent.set_env(self.env, 1, 1, 1, 1)  # is on [1,1] and will go to [1,1]

        agentDir = Direction.RIGHT

        self.iterationNo = 0

        for i in range(3):
            for x in range(2, 18):
                for y in range(2, 18):
                    print("Iteration:" + str(self.iterationNo))
                    self.agent.move(x,y)
                    self.SystemCalculate(self.agent.get_feature(Direction.UP), learning=True)

        expectedObject = [x[:] for x in [[0] * 20] * 20]

        A = [x[:] for x in [[0] * 20] * 20]
        B = [x[:] for x in [[0] * 20] * 20]

        predSDR1 = SDR(self.predictiveCellsSDR)
        predSDR2 = SDR(self.predictiveCellsSDR)

        # calculate what kind of object will system expect
        for x in range(2, 18):
            for y in range(2, 18):  # for sensor UP !
                self.agent.move(x, y)

                self.SystemCalculate("X", learning=False)
                scoreWithFeature = self.rawAnomaly

                self.SystemCalculate(" ", learning=False)
                scoreWithoutFeature = self.rawAnomaly

                # y -1 because we are using sensor UP
                A[x][y-1] = scoreWithFeature
                B[x][y-1] = scoreWithoutFeature
                expectedObject[x][y-1] = 1 if scoreWithFeature < scoreWithoutFeature else 0


        print(A)
        print(B)
        print(expectedObject)

        # Plotting and visualising environment-------------------------------------------
        if (
                fig_expect == None or isNotebook()
        ):  # create figure only if it doesn't exist yet or we are in interactive console
            fig_expect, _ = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
        else:
            fig_expect.axes[0].clear()

        plotBinaryMap(fig_expect.axes[0], "Expectation", expectedObject)
        fig_expect.canvas.draw()

        plt.show(block=True)
        #plt.pause(20)  # delay is needed for proper redraw

    def RunExperiment2(self):
        global fig_expect

        # put agent in the environment
        self.agent.set_env(self.env, 1, 1, 1, 1)  # is on [1,1] and will go to [1,1]

        self.iterationNo = 0
        random.seed = 42

        for i in range(1000):
            print("Iteration:" + str(self.iterationNo))
            self.SystemCalculate(self.agent.get_feature(Direction.UP), learning=True)
            # this tells agent where he will make movement next time & it will make previously requested movement
            self.agent.nextMove(random.randint(3,10), random.randint(3,10))


if __name__ == "__main__":

    # load model parameters from file
    f = open("modelParams.cfg", "r").read()
    parameters = eval(f)
    spParams = parameters["sensoryLayer_sp"]
    locParams = parameters["locationLayer"]
    tmParams = parameters["sensoryLayer_tm"]

    experiment = Experiment()
    # set up system
    experiment.SystemSetup()

    # initialize pandaBaker
    if PANDA_VIS_BAKE_DATA:
        experiment.BuildPandaSystem()


    experiment.RunExperiment1()


    if PANDA_VIS_BAKE_DATA:
        pandaBaker.CommitBatch()