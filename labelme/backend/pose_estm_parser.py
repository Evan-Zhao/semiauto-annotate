from qtpy.QtCore import QPointF

from labelme.shape import Shape, PoseShape


class PoseEstmParser(object):
    n_pose_points = 18
    body_chains = [
        [1, 2, 3, 4], [1, 5, 6, 7],
        [1, 8, 9, 10], [1, 11, 12, 13], [1, 0]
    ]
    keypoints_mapping = [
        'Nose', 'Neck',
        'R-Sho', 'R-Elb', 'R-Wr',
        'L-Sho', 'L-Elb', 'L-Wr',
        'R-Hip', 'R-Knee', 'R-Ank',
        'L-Hip', 'L-Knee', 'L-Ank',
        'R-Eye', 'L-Eye', 'R-Ear', 'L-Ear'
    ]
    shape_label = 'person'
    shape_form = [['person', None, None, None]]
    shape_type = 'linestrip'

    def __init__(self, json_data, accepted_label=None):
        self.data = None
        if accepted_label:
            self.accepted_label = {l: i for i, l in enumerate(accepted_label)}
        else:
            self.accepted_label = None
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

    def translate_label(self, i):
        if not self.accepted_label:
            return i
        name = self.keypoints_mapping[i]
        return self.accepted_label.get(name, None)

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
                translated_i = self.translate_label(i)
                if translated_i is None:
                    print(f'Dropped point of label {self.keypoints_mapping[i]}')
                else:
                    pose_parsed.append((QPointF(*p), translated_i))
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
