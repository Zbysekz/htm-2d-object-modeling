import yaml


class TwoDimensionalObjectSpace:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self._features = [x[:] for x in [[None] * height] * width]

    def load_object(self, yaml_text):
        obj = yaml.safe_load(yaml_text)

        if self.width < obj.get("width") or self.height < obj.get("height"):
            raise RuntimeError("Dimension of object is bigger than environment!")


        for x in range(self.width):
            for y in range(self.height):
                self._features[x][y] = None

        for feature in obj.get("features"):
            x = feature.get("x")
            y = feature.get("y")
            data = feature.get("data")

            if x >= obj.get("width") or y >= obj.get("height"):
                raise RuntimeError(
                    "Feature in object is outside the environment!" + str([x, y])
                )

            self._features[x][y] = data

    def size(self):
        return self.width * self.height

    def set_feature(self, x, y, f):
        if x < 0 or y < 0 or x >= self.width or y >= self.height:
            raise RuntimeError("Not possible to set features outside borders!")
        else:
            self._features[x][y] = f

    def get_feature(self, x, y):
        if (
            x < 0 or y < 0 or x >= self.width or y >= self.height
        ):  # return None for out of border positions
            return None

        return self._features[x][y]
