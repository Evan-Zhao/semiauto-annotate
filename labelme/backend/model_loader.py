from .yolo_parser import YoloParser


class ModelLoader(object):
    def __init__(self):
        self.yolo = YoloParser('yolo3/output.json')
        self.data = self.yolo.data
