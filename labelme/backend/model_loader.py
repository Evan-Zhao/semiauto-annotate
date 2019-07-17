from .pose_estm_parser import PoseEstmParser
from .yolo_parser import YoloParser

from time import sleep
from threading import Thread


class ModelLoader(object):
    @staticmethod
    def _load(image_file, on_completion):
        from labelme.utils import Config
        labels = Config.get('labels').keys()
        yolo = YoloParser(
            'yolo3/output.json',
            accepted_label=labels
        )
        pose_estm = PoseEstmParser('pose-estm/output.json')
        sleep(5)
        on_completion(yolo.data + pose_estm.data)

    @staticmethod
    def threaded_load(image_file, on_completion):
        load_thread = Thread(
            target=ModelLoader._load,
            args=(image_file, on_completion)
        )
        load_thread.start()
