import json
import copy
from htm.advanced.frameworks.location.path_integration_union_narrowing import \
  computeRatModuleParametersFromReadoutResolution
from htm.advanced.frameworks.location.path_integration_union_narrowing import computeRatModuleParametersFromCellCount

def createL4L6Nework(network, L4Params, L6aParams, inverseReadoutResolution=None, baselineCellsPerAxis=6,
                              suffix=""):
  """
    Create a single column network containing L4 and L6a layers. L4 layer
    processes sensor inputs while L6a processes motor commands using grid cell
    modules. Sensory input is represented by the feature's active columns and
    motor input is represented by the displacement vector [dx, dy].

    The grid cell modules used by this network are based on
    :class:`ThresholdedGaussian2DLocationModule` where the firing rate is computed
    from on one or more Gaussian activity bumps. The cells are distributed
    uniformly through the rhombus, packed in the optimal hexagonal arrangement.
    ::

    Phase
    -----                    +-------+
                 +---------->|       |<------------+
     [2]         |     +---->|  L4   |--winner---+ |
                 |     |     |       |           | |
                 |     |     +-------+           | |
                 |     |       |   ^             | |
                 |     |       |   |             | |
                 |     |       |   |             | |
                 |     |       v   |             | |
                 |     |     +-------+           | |
                 |     |     |       |           | |
     [1,3]       |     +---->|  L6a  |<----------+ |
                 |     |     |       |--learnable--+
                 |     |     +-------+
                 |     |         ^
            feature  reset       |
                 |     |         |
                 |     |         |
     [0]      [sensorInput] [motorInput]


    .. note::
        Region names are "motorInput", "sensorInput", "L4", and "L6a".
        Each name has an optional string suffix appended to it.

    :param network: network to add the column
    :type network: Network
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
  L6aParams = copy.deepcopy(L6aParams)

  numOfcols = L4Params["columnCount"]
  cellsPerCol = L4Params["cellsPerColumn"]
  L6aParams["anchorInputSize"] = numOfcols * cellsPerCol

  # Configure L4 'basalInputSize' to be compatible L6a output
  moduleCount = L6aParams["moduleCount"]
  cellsPerAxis = L6aParams["cellsPerAxis"]

  L4Params = copy.deepcopy(L4Params)
  L4Params["basalInputWidth"] = moduleCount * cellsPerAxis * cellsPerAxis

  # Configure sensor output to be compatible with L4 params
  columnCount = L4Params["columnCount"]

  # Add regions to network
  motorInputName = "motorInput" + suffix
  sensorInputName = "sensorInput" + suffix
  L4Name = "L4" + suffix
  L6aName = "L6a" + suffix
  dimensions = L6aParams.get("dimensions", 2)

  network.addRegion(sensorInputName, "py.RawSensor", json.dumps({"outputWidth": columnCount}))
  network.addRegion(motorInputName, "py.RawValues", json.dumps({"outputWidth": dimensions}))
  network.addRegion(L4Name, "py.ApicalTMPairRegion", json.dumps(L4Params))
  network.addRegion(L6aName, "py.GridCellLocationRegion", json.dumps(L6aParams))

  # Link sensory input to L4
  network.link(sensorInputName, L4Name, "UniformLink", "", srcOutput="dataOut", destInput="activeColumns")

  # Link motor input to L6a
  network.link(motorInputName, L6aName, "UniformLink", "", srcOutput="dataOut", destInput="displacement")

  # Link L6a to L4
  network.link(L6aName, L4Name, "UniformLink", "", srcOutput="activeCells", destInput="basalInput")
  network.link(L6aName, L4Name, "UniformLink", "", srcOutput="learnableCells", destInput="basalGrowthCandidates")

  # Link L4 feedback to L6a
  network.link(L4Name, L6aName, "UniformLink", "", srcOutput="activeCells", destInput="anchorInput")
  network.link(L4Name, L6aName, "UniformLink", "", srcOutput="winnerCells", destInput="anchorGrowthCandidates")

  # Link reset signal to L4 and L6a
  network.link(sensorInputName, L4Name, "UniformLink", "", srcOutput="resetOut", destInput="resetIn")
  network.link(sensorInputName, L6aName, "UniformLink", "", srcOutput="resetOut", destInput="resetIn")

  # Set phases appropriately
  network.setPhases(motorInputName, set([0]))
  network.setPhases(sensorInputName, set([0]))
  network.setPhases(L4Name, set([2]))
  network.setPhases(L6aName, set([1, 3]))

  return network