# Agent is entity with four sensors around him
from enum import Enum


class Direction(Enum):
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3


class Agent:
    def set_objectSpace(self, objectSpace, x, y, nextX=None, nextY=None):
        self._objSpace = objectSpace
        self._x = x
        self._y = y
        self._nextX = nextX if nextX is not None else x
        self._nextY = nextY if nextY is not None else y

    def get_feature(self, sensorLoc):
        if type(sensorLoc) != Direction:
            raise TypeError("Use enumeration Direction!")

        if sensorLoc == Direction.LEFT:
            f = self._objSpace.get_feature(self._x - 1, self._y)
        elif sensorLoc == Direction.RIGHT:
            f = self._objSpace.get_feature(self._x + 1, self._y)
        elif sensorLoc == Direction.UP:
            f = self._objSpace.get_feature(self._x, self._y - 1)
        elif sensorLoc == Direction.DOWN:
            f = self._objSpace.get_feature(self._x, self._y + 1)
        else:
            raise NotImplemented("Wrong SensorLoc!")
        return f

    def get_position(self):
        return [self._x, self._y]

    def get_nextPosition(self):
        return [self._nextX, self._nextY]

    def move(self, x, y):
        if x < 0 or y < 0 or x >= self._objSpace.width or y >= self._objSpace.height:
            raise RuntimeError(
                "Can't move outside environment borders!Pos:" + str([x, y])
            )
        self._x = x
        self._y = y

    def nextMove(self, x, y):# this tells agent where he will make movement next time & it will make previously requested movement

        self.move(self._nextX, self._nextY)

        self._nextX = x
        self._nextY = y

    def moveDir(self, direction):
        x = self._x
        y = self._y
        if direction == Direction.LEFT:
            x -= 1
        elif direction == Direction.RIGHT:
            x += 1
        elif direction == Direction.UP:
            y -= 1
        elif direction == Direction.DOWN:
            y += 1
        else:
            raise NotImplemented("Wrong direction!")

        self.move(x, y)

    def isBorderInThisDir(self, direction):
        x = self._x
        y = self._y
        if direction == Direction.LEFT:
            x -= 1
        elif direction == Direction.RIGHT:
            x += 1
        elif direction == Direction.UP:
            y -= 1
        elif direction == Direction.DOWN:
            y += 1
        else:
            raise NotImplemented("Wrong direction!")

        if x < 1 or y < 1 or x >= self._objSpace._width-1 or y >= self._objSpace._height-1:
            return True
        else:
            return False
