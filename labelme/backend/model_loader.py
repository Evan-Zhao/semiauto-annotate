from threading import Thread
from queue import Queue, Empty
from time import time

from labelme.utils import Config
import labelme.model
from pose_estm.pose_detection import PoseDetection
from yolo.yolo_class import YOLO
from .pose_estm_parser import PoseEstmParser
from .yolo_parser import YoloParser


class ModelLoader(object):
    exit_request = False

    def __init__(self):
        t0 = time()
        self.pose_detection = PoseDetection()
        t1 = time()
        print(f'Pose detection loaded in {t1 - t0} secs')
        self.yolo = YOLO()
        print(f'Yolo loaded in {time() - t1} secs')
        self.task_queue = Queue()

        results = labelme.model.get_unpreprocessed_img()
        for result in results:
            image_path = result['filename']
            img_id = result['id']
            self.task_queue.put((image_path, img_id, labelme.model.on_infer_complete))

        while not self.exit_request:
            try:
                image_file, img_id, on_completion = self.task_queue.get(timeout=1.0)
                self._infer(image_file, img_id, on_completion)
            except Empty:
                pass

    def _infer(self, image_file, img_id, on_completion):
        print("Start Inference")
        results = []
        t0 = time()
        try:
            labels = Config.get('labels').keys()
            yolo_json = self.yolo.infer_on_image(image_file)
            yolo = YoloParser(yolo_json, accepted_label=labels)
            results.extend(yolo.data)
        except Exception as e:
            print(e)
        t1 = time()
        print(f'Yolo inference in {t1 - t0} secs')
        try:
            point_labels = Config.get('point_labels')
            pose_estm_json = self.pose_detection.infer_on_image(image_file)
            pose_estm = PoseEstmParser(pose_estm_json, accepted_label=point_labels)
            results.extend(pose_estm.data)
        except Exception as e:
            print(e)
        print(f'Pose detection inference in {time() - t1} secs')
        on_completion([yolo_json, pose_estm_json], img_id, image_file)

    @classmethod
    def main_thread_ctor(cls):
        load_thread = Thread(target=cls)
        load_thread.start()

    '''
    def main_thread_infer(self, image_path, on_completion):
        self.task_queue.put((image_path, on_completion))
    '''
