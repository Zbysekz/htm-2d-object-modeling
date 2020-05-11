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

PLOT_GRAPHS = True
PLOT_ENV = True
PANDA_VIS_ENABLED = True

# Panda vis
if PANDA_VIS_ENABLED:
    from PandaVis.pandaComm.server import PandaServer
    from PandaVis.pandaComm.dataExchange import ServerData, dataHTMObject, dataLayer, dataInput

_EXEC_DIR = os.path.dirname(os.path.abspath(__file__))
# go one folder up and then into the objects folder
_OBJECTS_DIR = os.path.join(_EXEC_DIR, os.path.pardir, "objects")

OBJECT_FILENAME = "a.yml"  # what object to load

if PANDA_VIS_ENABLED:
    pandaServer = PandaServer()

class ObjectRecognitionExperiment:

    def __init__(self,parameters, verbose=True):

        self.parameters = parameters
        self.iterationNo = 0

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
        sp_info = Metrics(self.sensorLayer_sp.getColumnDimensions(), 999999999)

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

        self.locationLayer_SDR_cells = SDR(self.gridCellEncoder.dimensions)

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
        tm_info = Metrics([self.sensorLayer_tm.numberOfCells()], 999999999)

        self.predictiveCellsSDR = None
        self.anomalyHistData = []

        self.serverData = None

        self.fig_environment = None
        self.fig_graphs = None


    def SystemCalculate(self, feature, learning):

        # ENCODE DATA TO SDR--------------------------------------------------
        # Convert sensed feature to int
        self.sensedFeature = 1 if feature == "X" else 0
        self.sensorSDR = self.sensorEncoder.encode(self.sensedFeature)

        # ACTIVATE COLUMNS IN SENSORY LAYER ----------------------------------
        # Execute Spatial Pooling algorithm on Sensory Layer with sensorSDR as proximal input
        self.sensorLayer_sp.compute(self.sensorSDR, learning, self.sensorLayer_SDR_columns)

        if self.sensorLayer_sp.getIterationNum() == 1:
            rawAnomaly = 0
        else:
        # and calculate anomaly - compare how much of active columns had some predictive cells
            rawAnomaly = Anomaly.calculateRawAnomaly(self.sensorLayer_SDR_columns,
                                                 self.sensorLayer_tm.cellsToColumns(self.predictiveCellsSDR))

        # SIMULATE LOCATION LAYER --------------------------------------------
        # Execute Location Layer - it is just GC encoder
        self.gridCellEncoder.encode(self.agent.get_nextPosition(), self.locationLayer_SDR_cells)

        #
        # Execute Temporal memory algorithm over the Sensory Layer, with mix of
        # Location Layer activity and Sensory Layer activity as distal input
        externalDistalInput = self.locationLayer_SDR_cells


        if self.sensorLayer_sp.getIterationNum()==1:
            # activateDendrites calculates active segments - only for first time step here
            self.sensorLayer_tm.activateDendrites(learn=learning, externalPredictiveInputsActive=externalDistalInput,
                                                  externalPredictiveInputsWinners=externalDistalInput)

        # activates cells in columns by TM algorithm (winners, bursting...)
        self.sensorLayer_tm.activateCells(self.sensorLayer_SDR_columns, learning)


        print("Position:" + str(self.agent.get_position()))
        print("Feature:" + str(self.sensedFeature))
        print("Anomaly score:" + str(rawAnomaly))
        self.anomalyHistData += [rawAnomaly]
        # ------------------HTMpandaVis----------------------


        if PLOT_ENV and \
                (not pandaServer.cmdGotoIteration or (
                        pandaServer.cmdGotoIteration and pandaServer.gotoIteration == pandaServer.currentIteration+1)):
            # Plotting and visualising environment-------------------------------------------
            if (
                    self.fig_environment == None or isNotebook()
            ):  # create figure only if it doesn't exist yet or we are in interactive console
                self.fig_environment, _ = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
            else:
                self.fig_environment.axes[0].clear()

            plotEnvironment(self.fig_environment.axes[0], "Environment", self.env, self.agent.get_position())
            self.fig_environment.canvas.draw()

            plt.show(block=False)
            plt.pause(0.001)  # delay is needed for proper redraw

        if PANDA_VIS_ENABLED:
            # activateDendrites calculates active segments
            self.sensorLayer_tm.activateDendrites(learn=learning, externalPredictiveInputsActive=externalDistalInput,
                                                  externalPredictiveInputsWinners=externalDistalInput)

            self.predictiveCellsSDR = self.sensorLayer_tm.getPredictiveCells()
            self.PandaUpdateData()
            pandaServer.BlockExecution()


        if PLOT_GRAPHS and\
                (not pandaServer.cmdGotoIteration or (pandaServer.cmdGotoIteration and pandaServer.gotoIteration == pandaServer.currentIteration+1)):
            # ---------------------------
            if (
                    self.fig_graphs == None or isNotebook()
            ):  # create figure only if it doesn't exist yet or we are in interactive console
                self.fig_graphs, _ = plt.subplots(nrows=1, ncols=1, figsize=(5, 2))
            else:
                self.fig_graphs.axes[0].clear()

            self.fig_graphs.axes[0].set_title("Anomaly score")
            self.fig_graphs.axes[0].plot(self.anomalyHistData)
            self.fig_graphs.canvas.draw()

            #if agent.get_position() != [3, 4]:  # HACK ALERT! Ignore at this pos (after reset)
            #    anomalyHistData += [sensorLayer_tm.anomaly]




    def BuildPandaSystem(self):
        self.serverData = ServerData()
        self.serverData.HTMObjects["HTM1"] = dataHTMObject()
        self.serverData.HTMObjects["HTM1"].inputs["FeatureSensor"] = dataInput()

        self.serverData.HTMObjects["HTM1"].layers["SensoryLayer"] = dataLayer(
            modelParams["sensorLayer_sp"]["columnCount"],
            modelParams["sensorLayer_tm"]["cellsPerColumn"],
        )
        self.serverData.HTMObjects["HTM1"].layers["SensoryLayer"].proximalInputs = ["FeatureSensor"]
        self.serverData.HTMObjects["HTM1"].layers["SensoryLayer"].distalInputs = ["LocationLayer"]


        self.serverData.HTMObjects["HTM1"].inputs["LocationLayer"] = dataInput() # for now, Location layer is just position encoder

    def PandaUpdateData(self):
          # ------------------HTMpandaVis----------------------
          # fill up values
          pandaServer.currentIteration = self.sensorLayer_sp.getIterationNum()
          # do not update if we are running GOTO iteration command
          if (not pandaServer.cmdGotoIteration or (
                  pandaServer.cmdGotoIteration and pandaServer.gotoIteration == pandaServer.currentIteration)):

              self.serverData.iterationNo = pandaServer.currentIteration
              self.serverData.HTMObjects["HTM1"].inputs["FeatureSensor"].stringValue = "Feature: {:.2f}".format(self.sensedFeature)
              self.serverData.HTMObjects["HTM1"].inputs["FeatureSensor"].bits = self.sensorSDR.sparse
              self.serverData.HTMObjects["HTM1"].inputs["FeatureSensor"].count = self.sensorSDR.size

              self.serverData.HTMObjects["HTM1"].inputs["LocationLayer"].stringValue = str(self.agent.get_position())
              self.serverData.HTMObjects["HTM1"].inputs["LocationLayer"].bits = self.locationLayer_SDR_cells.sparse
              self.serverData.HTMObjects["HTM1"].inputs["LocationLayer"].count = self.locationLayer_SDR_cells.size

              self.serverData.HTMObjects["HTM1"].layers["SensoryLayer"].activeColumns = self.sensorLayer_SDR_columns.sparse

              self.serverData.HTMObjects["HTM1"].layers["SensoryLayer"].winnerCells = self.sensorLayer_tm.getWinnerCells().sparse
              self.serverData.HTMObjects["HTM1"].layers["SensoryLayer"].activeCells = self.sensorLayer_tm.getActiveCells().sparse
              self.serverData.HTMObjects["HTM1"].layers["SensoryLayer"].predictiveCells = self.predictiveCellsSDR.sparse

          # print("ACTIVECOLS:"+str(serverData.HTMObjects["HTM1"].layers["SensoryLayer"].activeColumns ))
          # print("WINNERCELLS:"+str(serverData.HTMObjects["HTM1"].layers["SensoryLayer"].winnerCells))
          # print("ACTIVECELLS:" + str(serverData.HTMObjects["HTM1"].layers["SensoryLayer"].activeCells))
          # print("PREDICTCELLS:"+str(serverData.HTMObjects["HTM1"].layers["SensoryLayer"].predictiveCells))

              pandaServer.serverData = self.serverData

              pandaServer.spatialPoolers["HTM1"] = self.sensorLayer_sp
              pandaServer.temporalMemories["HTM1"] = self.sensorLayer_tm
              pandaServer.NewStateDataReady()

if __name__ == "__main__":

    # load model parameters from file
    f = open("modelParams.cfg", "r").read()
    modelParams = eval(f)

    experiment = ObjectRecognitionExperiment(modelParams)

    if PANDA_VIS_ENABLED:
        # set up pandaVis
        pandaServer.Start()
        experiment.BuildPandaSystem()

    # put agent in the environment
    experiment.agent.set_env(experiment.env, 1, 1, 1, 1) # is on [1,1] and will go to [1,1]

    agentDir = Direction.RIGHT

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

    # iterationNo = 0
    # for i in range(10):
    #     for x in range(1, 19):
    #         for y in range(1, 19):
    #             print("Iteration:" + str(iterationNo))
    #
    #             if iterationNo <= 246:
    #                 pandaServer.runOneStep = True
    #             SystemCalculate(agent.get_feature(Direction.UP))
    #
    #             if iterationNo == 245:
    #                 print(serverData.HTMObjects["HTM1"].layers["SensoryLayer"].activeColumns)
    #                 print(serverData.HTMObjects["HTM1"].layers["SensoryLayer"].winnerCells)
    #                 print(serverData.HTMObjects["HTM1"].layers["SensoryLayer"].predictiveCells)
    #             if iterationNo == 246:
    #                 print(serverData.HTMObjects["HTM1"].layers["SensoryLayer"].activeColumns)
    #                 print(serverData.HTMObjects["HTM1"].layers["SensoryLayer"].winnerCells)
    #                 print(serverData.HTMObjects["HTM1"].layers["SensoryLayer"].predictiveCells)
    #
    #             agent.move(x, y)
    #
    #             iterationNo += 1

    iterationNo = 0
    # for i in range(100000):
    #     for x in range(1, 19):
    #         for y in range(1, 19):
    #             print("Iteration:" + str(iterationNo))
    #             SystemCalculate(agent.get_feature(Direction.UP))
    #
    #             agent.nextMove(x, y) # this tells agent where he will make movement next time & it will make previously requested movement
    #
    #             iterationNo += 1

    #predictiveCellsSDR_last = SDR( modelParams["sensorLayer_sp"]["columnCount"]*modelParams["sensorLayer_tm"]["cellsPerColumn"])

    while True:
        goal = [random.randrange(2, 8),random.randrange(2, 8)]
        print("goal is:"+str(goal))
        while experiment.agent.get_position() != goal:

            print("Iteration:" + str(iterationNo))
            experiment.SystemCalculate(experiment.agent.get_feature(Direction.UP),learning=True)

            pos = experiment.agent.get_position()# go by one step closer to goal
            if pos[0] > goal[0]:
                pos[0] -= 1
            elif pos[0] < goal[0]:
                pos[0] += 1
            elif pos[1] > goal[1]:
                pos[1] -= 1
            elif pos[1] < goal[1]:
                pos[1] += 1

            experiment.agent.nextMove(pos[0],
                                      pos[1])  # this tells agent where he will make movement next time & it will make previously requested movement

            iterationNo += 1

    # expectedObject = [x[:] for x in [[0] * 20] * 20]
    #
    # A = [x[:] for x in [[0] * 20] * 20]
    # B = [x[:] for x in [[0] * 20] * 20]
    #
    # predSDR1 = SDR(predictiveCellsSDR)
    # predSDR2 = SDR(predictiveCellsSDR)
    #
    # # calculate what kind of object will system expect
    # for x in range(0,20):
    #     for y in range(1,20):# for sensor UP !
    #         agent.nextMove(x, y)
    #
    #         SystemCalculate("X", learning=False, predictiveCellsSDR_last = predSDR1)
    #         predSDR1 = predictiveCellsSDR
    #         print("active:" + str(sensorLayer_SDR_columns.sparse))
    #         print("predictive:"+ str(predictiveCellsSDR))
    #         scoreWithFeature = rawAnomaly
    #
    #         SystemCalculate(" ", learning=False, predictiveCellsSDR_last = predSDR2)
    #         predSDR2 = predictiveCellsSDR
    #         print("active:" + str(sensorLayer_SDR_columns.sparse))
    #         print("predictive:" + str(predictiveCellsSDR))
    #         scoreWithoutFeature = rawAnomaly
    #
    #         A[x][y] = scoreWithFeature
    #         B[x][y] = scoreWithoutFeature
    #         expectedObject[x][y] = 1 if scoreWithFeature > scoreWithoutFeature else 0
    #
    #
    # print(A)
    # print(B)
    # print(expectedObject)

    # Plotting and visualising environment-------------------------------------------
    if (
            fig_expect == None or isNotebook()
    ):  # create figure only if it doesn't exist yet or we are in interactive console
        fig_expect, _ = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
    else:
        fig_expect.axes[0].clear()

    plotBinaryMap(fig_expect.axes[0], "Expectation", expectedObject)
    fig_expect.canvas.draw()

    plt.show(block=False)
    plt.pause(20)  # delay is needed for proper redraw


    # for x in range(2000):
    #     for i in range(5):
    #         print("Iteration:" + str(iterationNo))
    #         SystemCalculate()
    #         agent.moveDir(agentDir)
    #         if agent.get_position() == [3, 4]:
    #             sensorLayer_tm.reset()
    #             print("reset!")
    #         time.sleep(0.01)
    #         iterationNo += 1
    #     agentDir = Direction.RIGHT if agentDir == Direction.LEFT else Direction.LEFT

    if PANDA_VIS_ENABLED:
        pandaServer.MainThreadQuitted()