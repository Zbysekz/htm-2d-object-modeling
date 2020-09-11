import sys
import json
sys.path.append('/home/osboxes/HTM/HTMpandaVis')  # DELETE AFTER DEBUGGING pandaVis!!!
sys.path.append('/home/osboxes/HTM/HTMpandaVis/pandaBaker')  # DELETE AFTER DEBUGGING pandaVis!!!
from pandaNetwork import Network
import math
import numpy as np
from htm.advanced.support.register_regions import registerAllAdvancedRegions


# net = Network()
# net.addRegion("WTFregion", "TMRegion",
#               str(dict(columnCount=100,
#          cellsPerColumn=20,
#          activationThreshold=0,
#          )))
#
# net.initialize()
#
#

registerAllAdvancedRegions()

""" Creating network instance. """
config = """
        {network: [
            {addRegion: {name: "encoder", type: "RDSEEncoderRegion", params: {size: 1000, sparsity: 0.2, radius: 0.03, seed: 2019, noise: 0.01}}},
            {addRegion: {name: "sp", type: "SPRegion", params: {columnCount: 2048, globalInhibition: true}}},
            {addRegion: {name: "tm", type: "TMRegion", params: {cellsPerColumn: 8, orColumnOutputs: true}}},
            {addRegion: {name: "apicalTM", type: "py.ApicalTMPairRegion", params: {columnCount : 2048, basalInputWidth : 10, cellsPerColumn: 8, implementation: ApicalTiebreak}}},
            {addLink:   {src: "encoder.encoded", dest: "sp.bottomUpIn"}},
            {addLink:   {src: "sp.bottomUpOut", dest: "tm.bottomUpIn"}},
            {addLink:   {src: "sp.bottomUpOut", dest: "apicalTM.activeColumns"}}
        ]}"""
net = Network()
net.configure(config)

#  feed data to RDSE encoder via its "sensedValue" parameter.
net.getRegion('encoder').setParameterReal64('sensedValue', 100)
net.run(10)  # Execute iteration of the Network object


print(net.getRegion('tm'))
print(net.getRegion('apicalTM'))
#print(regs[2][1].getNodeType())
#ret1 = net.getRegion('tm').getConnections("ttt")
ret1 = net.getRegion('tm').getConnections("")
ret2 = net.getRegion('apicalTM').getConnections("")

cntWith = 0
cntWIthout = 0
for seg in range(ret1.numCells()):
  a = ret1.numSegments(seg)
  if a >0:
    cntWith +=1
  else:
    cntWIthout +=1
print(cntWith)
print(cntWIthout)



