import copy
from l2l4l6Framework.l2_l4_l6_networkFactory import createL246Nework

def createMultipleL246aNetwork(network, numberOfColumns, L2Params,
                                      L4Params, L6aParams,
                                      inverseReadoutResolution=None,
                                      baselineCellsPerAxis=6):
  """
    Create a network consisting of multiple columns. Each column contains one L2,
    one L4 and one L6a layers identical in structure to the network created by
    :func:`createL246aLocationColumn`. In addition all the L2 columns are fully
    connected to each other through their lateral inputs.
    ::

                            +----lateralInput--+
                            | +--------------+ |
                            | |       +1     | |
 Phase                      v |              v |
 -----                   +-------+         +-------+
                  reset  |       |         |       | reset
 [3]              +----->|  L2   |         |  L2   |<----+
                  |      |       |         |       |     |
                  |      +-------+         +-------+     |
                  |        |   ^             |   ^       |
                  |     +1 |   |          +1 |   |       |
                  |        |   |             |   |       |
                  |        v   |             v   |       |
                  |      +-------+         +-------+     |
 [2]        +----------->|       |         |       |<----------+
            |     |      |  L4   |         |  L4   |     |     |
            |     +----->|       |         |       |<----+     |
            |     |      +-------+         +-------+     |     |
            |     |        |   ^             |   ^       |     |
            |     |        |   |             |   |       |     |
            |     |        |   |             |   |       |     |
            |     |        v   |             v   |       |     |
            |     |      +-------+         +-------+     |     |
            |     |      |       |         |       |     |     |
 [1,3]      |     +----->|  L6a  |         |  L6a  |<----+     |
            |     |      |       |         |       |     |     |
            |     |      +-------+         +-------+     |     |
       feature  reset        ^                 ^      reset  feature
            |     |          |                 |         |     |
            |     |          |                 |         |     |
 [0]     [sensorInput]  [motorInput]      [motorInput] [sensorInput]

    .. note::
        Region names are "motorInput", "sensorInput". "L2", "L4", and "L6a".
        Each name has column number appended to it.
        For example: "sensorInput_0", "L2_1", "L6a_0" etc.

    :param network: network to add the column
    :type network: Network
    :param numberOfColumns: Number of columns to create
    :type numberOfColumns: int
    :param L2Params:    constructor parameters for :class:`ColumnPoolerRegion`
    :type L2Params: dict
    :param L4Params:    constructor parameters for :class:`ApicalTMPairRegion`
    :type L4Params: dict
    :param L6aParams:    constructor parameters for :class:`GridCellLocationRegion`
    :type L6aParams: dict
    :param inverseReadoutResolution: Optional readout resolution.
        The readout resolution specifies the diameter of the circle of phases in the
        rhombus encoded by a bump. See `createRatModuleFromReadoutResolution.
    :type inverseReadoutResolution: int
    :param baselineCellsPerAxis: The baselineCellsPerAxis implies the readout
        resolution of a grid cell module. If baselineCellsPerAxis=6, that implies
        that the readout resolution is approximately 1/3. If baselineCellsPerAxis=8,
        the readout resolution is approximately 1/4
    :type baselineCellsPerAxis: int or float
    :return: Reference to the given network
    :rtype: Network
    """
  L2Params = copy.deepcopy(L2Params)
  L4Params = copy.deepcopy(L4Params)
  L6aParams = copy.deepcopy(L6aParams)

  # Update L2 numOtherCorticalColumns parameter
  L2Params["numOtherCorticalColumns"] = numberOfColumns - 1

  for i in range(numberOfColumns):
    # Make sure random seed is different for each column
    L2Params["seed"] = L2Params.get("seed", 42) + i
    L4Params["seed"] = L4Params.get("seed", 42) + i
    L6aParams["seed"] = L6aParams.get("seed", 42) + i

    # Create column
    network = createL246Nework(network=network,
                                        L2Params=L2Params,
                                        L4Params=L4Params,
                                        L6aParams=L6aParams,
                                        inverseReadoutResolution=inverseReadoutResolution,
                                        baselineCellsPerAxis=baselineCellsPerAxis,
                                        suffix="_" + str(i))

  # Now connect the L2 columns laterally
  if numberOfColumns > 1:
    for i in range(numberOfColumns):
      src = str(i)
      for j in range(numberOfColumns):
        if i != j:
          dest = str(j)
          network.link("L2_" + src, "L2_" + dest, "UniformLink", "", srcOutput="feedForwardOutput",
                       destInput="lateralInput", propagationDelay=1)

  return network
