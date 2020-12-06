import numpy as np
import matplotlib.colors as Colors
import random

class GCM_1D:
    # size: number of cells
    # scale: while shifting, one must shift by "scale" value to get to the same place in this 1D GCM
    def __init__(self, size, scale):
        self.cellActivities = np.zeros(size)
        self.scale = scale
        self.inhibitKoef = 0.05
        self.exciteKoef = 0.5

        self.shiftRemainder = 0.0

        self.threshold = 0.6

    def Compute(self):
        self.cellActivities_new = self.cellActivities.copy()


        for current in range(len(self.cellActivities)):

            if self.cellActivities[current] == 0.0:
                continue # nothing to compute for current cell if activity is zero

            self.exciteKoefArray = [0.0] * len(self.cellActivities)
            # excite nearby
            mu = 0
            sigma = 1.0
            endValue = 0.01

            halfLen = (len(self.cellActivities)-1)/2

            for sign in [+1, -1]:  # for forward and backward
                shift = 0
                while(shift < halfLen):  # go till half of len, but terminate sooner, as you reach small value "endValue"
                    shiftIdx = current + sign*shift
                    if sign>0:
                        if shiftIdx >= len(self.exciteKoefArray):
                            shiftIdx -= len(self.exciteKoefArray)
                    # if negative sign, use benefit of python negative indexes array[-1] = last item

                    self.exciteKoefArray[shiftIdx] =\
                        1/(sigma * np.sqrt(2 * np.pi)) *\
                        np.exp( - (shift - mu)**2 / (2 * sigma**2))

                    if self.exciteKoefArray[shiftIdx] <= endValue:
                        break

                    shift += 1


            for other in range(len(self.cellActivities)):
                if current != other:

                    # excitation
                    self.cellActivities_new[other] += self.cellActivities[current] * self.exciteKoefArray[other] * self.exciteKoef

                    # inhibit all other
                    self.cellActivities_new[other] -= self.cellActivities[current] * self.inhibitKoef
                    if self.cellActivities_new[other] < 0.0:
                        self.cellActivities_new[other] = 0.0

                    if self.cellActivities_new[other] > 1.0:
                        self.cellActivities_new[other] = 1.0

            self.cellActivities = self.cellActivities_new



    def Shift(self, value):
        # shift by "value", keep remainder value in internal storage
        self.shiftRemainder += value
        nOfCellShift = int(self.shiftRemainder // self.scale) # by how many cells to shift
        self.shiftRemainder = self.shiftRemainder % self.scale
        self.cellActivities = np.roll(self.cellActivities, nOfCellShift)

    def getSDR(self):
        return [x for x in self.cellActivities if x >= self.threshold]

    def InitRandomPos(self):
        self.cellActivities[random.randint(0, len(self.cellActivities)-1)] = 1.0

    def plot(self,
            axes
    ):

        axes.get_xaxis().set_visible(False)
        axes.get_yaxis().set_visible(False)

        axes.imshow(self.cellActivities[np.newaxis, :], interpolation="nearest",  cmap="hot") #norm=Colors.Normalize(0, 5),




