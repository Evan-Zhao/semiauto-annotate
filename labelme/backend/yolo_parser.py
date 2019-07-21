from qtpy.QtCore import QPointF

from labelme.shape import Shape


class YoloParser(object):
    def __init__(self, json_data, accepted_label=None):
        self.data = None
        self.accepted_label = accepted_label
        if json_data is not None:
            self.load(json_data)

    def load(self, json_data):
        import json
        from collections import namedtuple
        loaded = json.loads(
            json_data, object_hook=lambda d: namedtuple('X', d.keys(), rename=True)(*d.values())
        )
        self.data = [self.yolo_dict_to_shape(yolo_dict) for yolo_dict in loaded]
        self.data = [x for x in self.data if x is not None]

    def yolo_dict_to_shape(self, val):
        label = val._1
        if label not in self.accepted_label:
            print(f'Dropped shape of label {label}')
            return None
        loc = val.location
        points = [
            QPointF(loc.left, loc.top),
            QPointF(loc.right, loc.bottom)
        ]
        return Shape(
            points,
            form=[[label, None, None]],
            shape_type='rectangle'
        )
