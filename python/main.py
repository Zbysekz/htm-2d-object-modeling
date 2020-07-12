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
PANDA_VIS_BAKE_DATA = False # if we want to bake data for pandaVis tool (repo at https://github.com/htm-community/HTMpandaVis )

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

    def SystemSetup(self,parameters, verbose=True):

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
        spParams = parameters["sensorLayer_sp"]
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
        locParams = parameters["locationLayer"]

        self.gridCellEncoder = GridCellEncoder(
            size=locParams["cellCount"],
            sparsity=locParams["sparsity"],
            periods=locParams["periods"],
            seed=locParams["seed"],
        )

        self.locationlayer_SDR_cells = SDR(self.gridCellEncoder.dimensions)

        tmParams = parameters["sensorLayer_tm"]
        self.sensorLayer_tm = TemporalMemory(
            columnDimensions=(spParams["columnCount"],),
            cellsPerColumn=tmParams["cellsPerColumn"],
            activationThreshold=tmParams["activationThreshold"],
            initialPermanence=tmParams["initialPerm"],
            connectedPermanence=spParams["synPermConnected"],
            minThreshold=tmParams["minThreshold"],
            maxNewSynapseCount=tmParams["newSynapseCount"],
            permanenceIncrement=tmParams["permanenceInc"],
            permanenceDecrement=tmParams["permanenceDec"],
            predictedSegmentDecrement=0.0,
            maxSegmentsPerCell=tmParams["maxSegmentsPerCell"],
            maxSynapsesPerSegment=tmParams["maxSynapsesPerSegment"],
            externalPredictiveInputs=locParams["cellCount"],
        )
        self.tm_info = Metrics([self.sensorLayer_tm.numberOfCells()], 999999999)


    def SystemCalculate(self, feature, learning , predictiveCellsSDR_last):
        global fig_environment, fig_graphs
        # ENCODE DATA TO SDR--------------------------------------------------
        # Convert sensed feature to int
        self.sensedFeature = 1 if feature == "X" else 0
        self.sensorSDR = self.sensorEncoder.encode(self.sensedFeature)

        # ACTIVATE COLUMNS IN SENSORY LAYER ----------------------------------
        # Execute Spatial Pooling algorithm on Sensory Layer with sensorSDR as proximal input
        self.sensorLayer_sp.compute(self.sensorSDR, learning, self.sensorLayer_SDR_columns)

        if self.iterationNo!=0:
        # and calculate anomaly - compare how much of active columns had some predictive cells
            self.rawAnomaly = Anomaly.calculateRawAnomaly(self.sensorLayer_SDR_columns,
                                                 self.sensorLayer_tm.cellsToColumns(predictiveCellsSDR_last))
        else:
            self.rawAnomaly = 0

        # SIMULATE LOCATION LAYER --------------------------------------------
        # Execute Location Layer - it is just GC encoder
        self.gridCellEncoder.encode(self.agent.get_nextPosition(), self.locationlayer_SDR_cells)

        #
        # Execute Temporal memory algorithm over the Sensory Layer, with mix of
        # Location Layer activity and Sensory Layer activity as distal input
        externalDistalInput = self.locationlayer_SDR_cells



        self.sensorLayer_tm.activateCells(self.sensorLayer_SDR_columns, learning)

        # activateDendrites calculates active segments
        self.sensorLayer_tm.activateDendrites(learn=learning, externalPredictiveInputsActive=externalDistalInput,
                                         externalPredictiveInputsWinners=externalDistalInput)
        # predictive cells are calculated directly from active segments
        self.predictiveCellsSDR = self.sensorLayer_tm.getPredictiveCells()

        # PANDA VIS
        if PANDA_VIS_BAKE_DATA:
            # ------------------HTMpandaVis----------------------
            # fill up values
            pandaBaker.inputs["FeatureSensor"].stringValue = "Feature: {:.2f}".format(self.sensedFeature)
            pandaBaker.inputs["FeatureSensor"].bits = self.sensorSDR.sparse

            pandaBaker.inputs["LocationLayer"].stringValue = str(self.agent.get_position())
            pandaBaker.inputs["LocationLayer"].bits = self.locationlayer_SDR_cells.sparse

            pandaBaker.layers["SensoryLayer"].activeColumns = self.sensorLayer_SDR_columns.sparse
            pandaBaker.layers["SensoryLayer"].winnerCells =  self.sensorLayer_tm.getWinnerCells().sparse
            pandaBaker.layers["SensoryLayer"].predictiveCells = self.predictiveCellsSDR.sparse
            pandaBaker.layers["SensoryLayer"].activeCells = self.sensorLayer_tm.getActiveCells().sparse

            # customizable datastreams to be show on the DASH PLOTS
            pandaBaker.dataStreams["rawAnomaly"].value = self.rawAnomaly
            pandaBaker.dataStreams["numberOfWinnerCells"].value = len(self.sensorLayer_tm.getWinnerCells().sparse)
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
            print("ITERATION: " + str(self.iterationNo))

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
            #    anomalyHistData += [sensorLayer_tm.anomaly]




    def BuildPandaSystem(self):

        pandaBaker.inputs["FeatureSensor"] = cInput(self.sensorEncoder.size)

        pandaBaker.layers["SensoryLayer"] = cLayer(self.sensorLayer_sp, self.sensorLayer_tm)
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

        predictiveCellsSDR_last = SDR(
            modelParams["sensorLayer_sp"]["columnCount"] * modelParams["sensorLayer_tm"]["cellsPerColumn"])
        for i in range(1):
            for x in range(1, 19):
                for y in range(1, 19):
                    print("Iteration:" + str(self.iterationNo))
                    self.SystemCalculate(self.agent.get_feature(Direction.UP), learning=True,
                                               predictiveCellsSDR_last=predictiveCellsSDR_last)
                    predictiveCellsSDR_last = self.predictiveCellsSDR
                    self.agent.nextMove(x,
                                              y)  # this tells agent where he will make movement next time & it will make previously requested movement

        expectedObject = [x[:] for x in [[0] * 20] * 20]

        A = [x[:] for x in [[0] * 20] * 20]
        B = [x[:] for x in [[0] * 20] * 20]

        predSDR1 = SDR(self.predictiveCellsSDR)
        predSDR2 = SDR(self.predictiveCellsSDR)

        # calculate what kind of object will system expect
        for x in range(0, 19):
            for y in range(1, 20):  # for sensor UP !
                self.agent.nextMove(x, y)

                self.SystemCalculate("X", learning=False, predictiveCellsSDR_last=predSDR1)
                predSDR1 = self.predictiveCellsSDR
                print("active:" + str(self.sensorLayer_SDR_columns.sparse))
                print("predictive:" + str(self.predictiveCellsSDR))
                scoreWithFeature = self.rawAnomaly

                self.SystemCalculate(" ", learning=False, predictiveCellsSDR_last=predSDR2)
                predSDR2 = self.predictiveCellsSDR
                print("active:" + str(self.sensorLayer_SDR_columns.sparse))
                print("predictive:" + str(self.predictiveCellsSDR))
                scoreWithoutFeature = self.rawAnomaly

                A[x][y] = scoreWithFeature
                B[x][y] = scoreWithoutFeature
                expectedObject[x][y] = 1 if scoreWithFeature > scoreWithoutFeature else 0


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
        random.seed(1)

        # for x in range(2000):
        #     print("Iteration:" + str(iterationNo))
        #     SystemCalculate(agent.get_feature(Direction.UP))
        #
        #     # find direction that is not behind border of environment
        #     agentDir = Direction(random.randrange(0, 4))
        #     while agent.isBorderInThisDir(agentDir):
        #         agentDir = Direction(random.randrange(0, 4))
        #
        #     agent.moveDir(agentDir)
        #
        #     if PLOT_ENV or PLOT_GRAPHS:
        #         time.sleep(0.01)
        #     iterationNo += 1

if __name__ == "__main__":

    # load model parameters from file
    f = open("modelParams.cfg", "r").read()
    modelParams = eval(f)

    experiment = Experiment()
    # set up system
    experiment.SystemSetup(modelParams)

    # initialize pandaBaker
    if PANDA_VIS_BAKE_DATA:
        experiment.BuildPandaSystem()


    experiment.RunExperiment1()


    if PANDA_VIS_BAKE_DATA:
        pandaBaker.CommitBatch()