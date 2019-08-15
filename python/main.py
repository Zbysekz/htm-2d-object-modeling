#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os.path
import yaml
import math
import htm2d.environment
import htm2d.agent
from htm2d.agent import Direction
import numpy as np
import time
import matplotlib.pyplot as plt
  
from htm.bindings.algorithms import SpatialPooler
from htm.bindings.sdr import SDR, Metrics
from htm.encoders.rdse import RDSE, RDSE_Parameters
from htm.encoders.grid_cell_encoder import GridCellEncoder

_EXEC_DIR = os.path.dirname(os.path.abspath(__file__))
#go one folder up and then into the objects folder
_OBJECTS_DIR=os.path.join(_EXEC_DIR,os.path.pardir, 'objects')

OBJECT_FILENAME = 'a.yml'#what object to load



def SystemSetup(parameters,verbose=True):
  global agent, sensorEncoder, env, sensorLayer_sp, sensorLayer_SDR_columns
  global gridCellEncoder,locationlayer_SDR_cells

  if verbose:
    import pprint
    print("Parameters:")
    pprint.pprint(parameters, indent=4)
    print("")
    
  # create environment and the agent
  env = htm2d.environment.TwoDimensionalEnvironment(20, 20)
  agent = htm2d.agent.Agent()
    
    
  # load object from yml file
  with open(os.path.join(_OBJECTS_DIR,OBJECT_FILENAME), 'r') as stream:
    try:
      env.load_object(stream)
    except yaml.YAMLError as exc:
      print(exc)
            
  # SENSOR LAYER --------------------------------------------------------------
  #setup sensor encoder
  sensorEncoderParams            = RDSE_Parameters()
  sensorEncoderParams.category   = True
  sensorEncoderParams.size       = parameters["enc"]["size"]
  sensorEncoderParams.sparsity = parameters["enc"]["sparsity"]
  sensorEncoder = RDSE( sensorEncoderParams )
  
  
  # Make the HTM.  SpatialPooler & TemporalMemory & associated tools.
  spParams = parameters["sensorLayer_sp"]
  sensorLayer_sp = SpatialPooler(
    inputDimensions            = (sensorEncoder.size,),
    columnDimensions           = (spParams["columnCount"],),
    potentialPct               = spParams["potentialPct"],
    potentialRadius            = sensorEncoder.size,
    globalInhibition           = True,
    localAreaDensity           = spParams["localAreaDensity"],
    synPermInactiveDec         = spParams["synPermInactiveDec"],
    synPermActiveInc           = spParams["synPermActiveInc"],
    synPermConnected           = spParams["synPermConnected"],
    boostStrength              = spParams["boostStrength"],
    wrapAround                 = True
  )
  sp_info = Metrics(sensorLayer_sp.getColumnDimensions(), 999999999 )
  
  # Create an SDR to represent active columns, This will be populated by the
  # compute method below. It must have the same dimensions as the Spatial Pooler.
  sensorLayer_SDR_columns = SDR( spParams["columnCount"] )
  
  # LOCATION LAYER ------------------------------------------------------------
  # Grid cell modules
  locParams = parameters["locationLayer"]
  
  gridCellEncoder = GridCellEncoder(
        size     = 100,
        sparsity = locParams["sparsity"],
        periods  = locParams["periods"])

  locationlayer_SDR_cells = SDR( gridCellEncoder.dimensions )
    
  #  tmParams = parameters["tm"]
  #  tm = TemporalMemory(
  #    columnDimensions          = (spParams["columnCount"],),
  #    cellsPerColumn            = tmParams["cellsPerColumn"],
  #    activationThreshold       = tmParams["activationThreshold"],
  #    initialPermanence         = tmParams["initialPerm"],
  #    connectedPermanence       = spParams["synPermConnected"],
  #    minThreshold              = tmParams["minThreshold"],
  #    maxNewSynapseCount        = tmParams["newSynapseCount"],
  #    permanenceIncrement       = tmParams["permanenceInc"],
  #    permanenceDecrement       = tmParams["permanenceDec"],
  #    predictedSegmentDecrement = 0.0,
  #    maxSegmentsPerCell        = tmParams["maxSegmentsPerCell"],
  #    maxSynapsesPerSegment     = tmParams["maxSynapsesPerSegment"]
  #  )
  #  tm_info = Metrics( [tm.numberOfCells()], 999999999 )


def SystemCalculate():
  global sensorLayer_sp
  
  # encode sensor data to SDR--------------------------------------------------
 
  # convert sensed feature to int
  sensedFeature = 1 if agent.get_feature(Direction.UP)=='X'else 0
  
  sensorSDR = sensorEncoder.encode(sensedFeature)
  
  print("Position:"+str(agent.get_position()))
  print("Feature:"+str(sensedFeature))
  print(sensorSDR)
  
  # put SDR to proximal input of sensorLayer-----------------------------------
  # sensorLayer.proximal = sensorSDR

  # Execute Spatial Pooling algorithm over input space.
  sensorLayer_sp.compute(sensorSDR, True, sensorLayer_SDR_columns)

  gridCellEncoder.encode(agent.get_position(), locationlayer_SDR_cells)
  

  plotBinaryMap("Input SDR", sensorSDR.size, sensorSDR.dense, subplot=131)
  
  plotBinaryMap("Sensor layer columns activation", sensorLayer_sp.getColumnDimensions()[0], sensorLayer_SDR_columns.dense, subplot=132)
  
  plotBinaryMap("Location layer cells activation", locationlayer_SDR_cells.size, locationlayer_SDR_cells.dense, subplot=133, drawPlot=True)
  
  
def plotBinaryMap(name, size, data, subplot=0, drawPlot=False):
  plotW = math.ceil(math.sqrt(size))

  rf = np.zeros([ plotW, plotW ], dtype=np.uint8)
  for i in range(plotW):
    arr = data[i*plotW:i*plotW+plotW]*2
    if len(arr)<plotW:
      arr=np.concatenate([arr, np.ones(plotW-len(arr))])
    rf[:, i] = arr
  
  if subplot>0:
    plt.subplot(subplot)
  
  plt.imshow(rf, interpolation='nearest')
  plt.title( name)
  plt.ylabel("Rows")
  plt.xlabel("Columns")

  if subplot>0:
    plt.tight_layout()
    
  if subplot==0 or subplot>0 and drawPlot:#if we are doing multiplot, draw only at the last call
    plt.show()

if __name__ == "__main__":
   
  # load model parameters from file
  f = open('modelParams.cfg','r').read()
  modelParams = eval(f)
  
  # set up system
  SystemSetup(modelParams)
  
  # put agent in the environment
  agent.set_env(env,3,4)
  
  agentDir = Direction.RIGHT
  
  for x in range(20):
    for i in range(5):
      print("Iteration:"+str(i))
      SystemCalculate()
      agent.moveDir(agentDir)
      time.sleep(0.3)
    agentDir = Direction.RIGHT if agentDir==Direction.LEFT else Direction.LEFT
  
    
    








