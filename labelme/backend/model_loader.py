from .yolo_parser import YoloParser
from .pose_estm_parser import PoseEstmParser


class ModelLoader(object):
    def __init__(self):
        from labelme.utils import Config

        labels = Config.get('labels').keys()
        self.yolo = YoloParser(
            'yolo3/output.json',
            accepted_label=labels
        )
        self.pose_estm = PoseEstmParser('pose-estm/output.json')
        self.data = self.yolo.data + self.pose_estm.data
