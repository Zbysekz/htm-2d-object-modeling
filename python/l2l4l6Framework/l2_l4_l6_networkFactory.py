import json
import copy
from l2l4l6Framework.l4_l6_networkFactory import createL4L6Nework

def createL246Nework(network, L2Params, L4Params, L6aParams,
                              baselineCellsPerAxis=6,
                              inverseReadoutResolution=None, suffix=""):
  """
    Create a single column network composed of L2, L4 and L6a layers.
    L2 layer computes the object representation using :class:`ColumnPoolerRegion`,
    L4 layer processes sensors input while L6a processes motor commands using grid
    cell modules. Sensory input is represented by the feature's active columns and
    motor input is represented by the displacement vector [dx, dy].

    The grid cell modules used by this network are based on
    :class:`ThresholdedGaussian2DLocationModule` where the firing rate is computed
    from on one or more Gaussian activity bumps. The cells are distributed
    uniformly through the rhombus, packed in the optimal hexagonal arrangement.
    ::

   Phase                       +-------+
   -----                reset  |       |
                        +----->|  L2   |<------------------+
   [3]                  |      |       |                   |
                        |      +-------+                   |
                        |        |   ^                     |
                        |        |   |                     |
                        |     +1 |   |                     |
                        |        v   |                     |
                        |      +-------+                   |
                  +----------->|       |--winnerCells------+
   [2]            |     |      |  L4   |<------------+
                  |     +----->|       |--winner---+ |
                  |     |      +-------+           | |
                  |     |        |   ^             | |
                  |     |        |   |             | |
                  |     |        |   |             | |
                  |     |        v   |             | |
                  |     |      +-------+           | |
                  |     |      |       |           | |
    [1,3]         |     +----->|  L6a  |<----------+ |
                  |     |      |       |--learnable--+
                  |     |      +-------+
             feature  reset        ^
                  |     |          |
                  |     |          |
    [0]        [sensorInput]  [motorInput]


    .. note::
        Region names are "motorInput", "sensorInput". "L2", "L4", and "L6a".
        Each name has an optional string suffix appended to it.

    :param network: network to add the column
    :type network: Network
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
    :param suffix: optional string suffix appended to region name. Useful when
                                 creating multicolumn networks.
    :type suffix: str
    :return: Reference to the given network
    :rtype: Network
    """

  # Configure L2 'inputWidth' to be compatible with L4
  numOfcols = L4Params["columnCount"]
  cellsPerCol = L4Params["cellsPerColumn"]
  L2Params = copy.deepcopy(L2Params)
  L2Params["inputWidth"] = numOfcols * cellsPerCol

  # Configure L4 'apicalInputWidth' to be compatible L2 output
  L4Params = copy.deepcopy(L4Params)
  L4Params["apicalInputWidth"] = L2Params["cellCount"]

  # Add L4 - L6a location layers
  network = createL4L6Nework(network=network,
                                      L4Params=L4Params,
                                      L6aParams=L6aParams,
                                      inverseReadoutResolution=inverseReadoutResolution,
                                      baselineCellsPerAxis=baselineCellsPerAxis,
                                      suffix=suffix)
  L4Name = "L4" + suffix
  sensorInputName = "sensorInput" + suffix

  # Add L2 - L4 object layers
  L2Name = "L2" + suffix
  network.addRegion(L2Name, "py.ColumnPoolerRegion", json.dumps(L2Params))

  # Link L4 to L2
  network.link(L4Name, L2Name, "UniformLink", "", srcOutput="activeCells", destInput="feedforwardInput")
  network.link(L4Name, L2Name, "UniformLink", "", srcOutput="winnerCells", destInput="feedforwardGrowthCandidates")

  # Link L2 feedback to L4
  network.link(L2Name, L4Name, "UniformLink", "", srcOutput="feedForwardOutput", destInput="apicalInput",
               propagationDelay=1)

  # Link reset output to L2
  network.link(sensorInputName, L2Name, "UniformLink", "", srcOutput="resetOut", destInput="resetIn")

  # Set L2 phase to be after L4
  network.setPhases(L2Name, set([3]))

  return network