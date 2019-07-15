from qtpy.QtCore import QPointF

from labelme.shape import Shape, LabeledPoint


class YoloParser(object):
    def __init__(self, filename, label_accepted=None):
        self.data = None
        self.label_accepted = label_accepted
        if filename is not None:
            self.load(filename)
        self.filename = filename

    def load(self, filename):
        import json
        from collections import namedtuple
        with open(filename, 'r') as f:
            loaded = json.load(f, object_hook=lambda d: namedtuple('X', d.keys(), rename=True)(*d.values()))
        self.data = [self.yolo_dict_to_shape(yolo_dict) for yolo_dict in loaded]

    @staticmethod
    def yolo_dict_to_shape(val):
        loc = val.location
        points = [
            QPointF(loc.left, loc.top),
            QPointF(loc.right, loc.top),
            QPointF(loc.right, loc.bottom),
            QPointF(loc.left, loc.bottom)
        ]
        points = [LabeledPoint(p) for p in points]
        return Shape.from_points(
            points,
            label=val._1,
            shape_type='rectangle'
        )
