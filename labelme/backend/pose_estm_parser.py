from qtpy.QtCore import QPointF

from labelme.shape import PoseShape


class PoseEstmParser(object):
    def __init__(self, json_data, accepted_label=None):
        self.data = []
        if accepted_label:
            self.accepted_label = {l: i for i, l in enumerate(accepted_label)}
        else:
            self.accepted_label = None
        if json_data is not None:
            self.load(json_data)

    def load(self, json_data):
        import json
        loaded = json.loads(json_data)
        self.assert_and_raise(type(loaded) is list)
        for pose in loaded:
            pose_parsed = []
            self.assert_and_raise(type(pose) is list)
            self.assert_and_raise(len(pose) == PoseShape.n_pose_points)
            for i, p in enumerate(pose):
                if p == -1:
                    pose_parsed.append(None)
                else:
                    self.assert_and_raise(type(p) is list and len(p) == 2)
                    pose_parsed.append(QPointF(*p))
            self.data.append(PoseShape(pose_parsed))

    @staticmethod
    def assert_and_raise(condition, msg=''):
        if not condition:
            raise ValueError(msg)
