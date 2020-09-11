from htm.bindings.engine_internal import Network
from htm.advanced.support.register_regions import registerAllAdvancedRegions

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

print(net.getRegion('tm').getConnections("")) # can be called because of this draft PR

print(net.getRegion('apicalTM').getConnections("")) # returns always None, it is region implemented in python, but it has not override getConnections

print(net.getRegion('tm').getAlgorithmInstance()) # cannot call this, not accessible



