from labelme.shape import Shape, MultiShape, LabeledPoint


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

    def __init__(self, filename):
        self.data = None
        if filename is not None:
            self.load(filename)
        self.filename = filename

    def load(self, filename):
        import json
        with open(filename, 'r') as f:
            loaded = json.load(f)
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
                pose_parsed.append(LabeledPoint(x=p[0], y=p[1], label=i))
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

        shapes = []
        for chain in all_chain_points:
            shapes.append(Shape.from_points(
                chain,
                label=PoseEstmParser.shape_label,
                shape_type=PoseEstmParser.shape_type
            ))
        multi_shape = MultiShape(shapes)
        return multi_shape
