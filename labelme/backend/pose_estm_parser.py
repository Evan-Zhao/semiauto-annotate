from qtpy.QtCore import QPointF

from labelme.shape import Shape, PoseShape


class PoseEstmParser(object):
    n_pose_points = 18
    body_chains = [
        [1, 8, 9, 10],
        [4, 3, 2, 16, 14, 0, 15, 17, 5, 6, 7],
        [13, 12, 11]
    ]
    shape_label = 'person'
    shape_form = [['person', None, None, None]]
    shape_type = 'linestrip'

    def __init__(self, json_data):
        self.data = None
        if json_data is not None:
            self.load(json_data)

    def load(self, json_data):
        import json
        loaded = json.loads(json_data)
        parsed = self.validate_and_parse(loaded)
        self.data = [self.points_to_shape(points) for points in parsed]

    @staticmethod
    def assert_and_raise(condition, msg=''):
        if not condition:
            raise ValueError(msg)

    def validate_and_parse(self, loaded):
        ret = []
        self.assert_and_raise(type(loaded) is list)
        for pose in loaded:
            pose_parsed = []
            self.assert_and_raise(type(pose) is list)
            self.assert_and_raise(len(pose) == self.n_pose_points)
            for i, p in enumerate(pose):
                if p == -1:
                    pose_parsed.append(None)
                    continue
                self.assert_and_raise(type(p) is list and len(p) == 2)
                pose_parsed.append((QPointF(*p), i))
            ret.append(pose_parsed)
        return ret

    @staticmethod
    def points_to_shape(points):
        def split_list_at_none(lst):
            last = 0
            for i, item in enumerate(lst):
                if item is None:
                    yield lst[last:i]
                    last = i + 1
            yield lst[last:]

        all_chain_points = []
        for chain in PoseEstmParser.body_chains:
            chain_points = [points[i] for i in chain]
            actual_chain = list(split_list_at_none(chain_points))
            actual_chain = [lst for lst in actual_chain if lst]
            all_chain_points.extend(actual_chain)

        shapes, annotation = [], []
        for chain in all_chain_points:
            points, labels = list(zip(*chain))
            shapes.append(Shape(
                points, form=None,
                shape_type=PoseEstmParser.shape_type
            ))
            annotation.append(labels)
        multi_shape = PoseShape(shapes, annotation=annotation)
        return multi_shape
