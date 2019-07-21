# -*- coding: utf-8 -*-
"""
Class definition of YOLO_v3 style detection model on image and video
"""

import colorsys
from timeit import default_timer as timer

import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw


from yolo.yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo.yolo3.utils import letterbox_image
import os
from keras.utils import multi_gpu_model

here = os.path.abspath(os.path.dirname(__file__))


class YOLO(object):
    _defaults = {
        "model_path": os.path.join(here, 'model_data/yolo.h5'),
        "anchors_path": os.path.join(here, 'model_data/yolo_anchors.txt'),
        "classes_path": os.path.join(here, 'model_data/coco_classes.txt'),
        "score": 0.3,
        "iou": 0.45,
        "model_image_size": (416, 416),
        "gpu_num": 1,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        import tensorflow.compat.v1.logging as logging
        import os
        logging.set_verbosity(logging.ERROR)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        self.__dict__.update(self._defaults)  # set up default values
        self.__dict__.update(kwargs)  # and update with user overrides
        self.image = True
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors == 6  # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None, None, 3)), num_anchors // 2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None, None, 3)), num_anchors // 3, num_classes)
            self.yolo_model.load_weights(self.model_path)  # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                   num_anchors / len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2,))
        if self.gpu_num >= 2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                                           len(self.class_names), self.input_image_shape,
                                           score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image):
        start = timer()

        if self.model_image_size != (None, None):
            assert self.model_image_size[0] % 32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1] % 32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        # print('Found {} boxes for {}'.format(len(out_boxes), 'img'))
        result = []
        for i in range(out_classes.shape[0]):
            top, left, bottom, right = out_boxes[i]
            top = max(0, int(np.floor(top + 0.5)))
            left = max(0, int(np.floor(left + 0.5).astype('int')))
            bottom = min(image.size[1], int(np.floor(bottom + 0.5).astype('int')))
            right = min(image.size[0], int(np.floor(right + 0.5).astype('int')))
            box_location = {"top": top, "left": left, "right": right, "bottom": bottom}
            class_name = self.class_names[out_classes[i]]
            score = float(out_scores[i])
            box = {"location": box_location, "class": class_name, "score": score}
            result.append(box)

        '''
        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

        end = timer()
        print(end - start)
        image.show()
        '''
        return result

    def infer_on_image(self, image_path):
        import json

        image = Image.open(image_path)
        result = self.detect_image(image)
        result = test_time_augmentation(self, image, result)

        # self.close_session()
        return json.dumps(result)

    def close_session(self):
        self.sess.close()


def detect_video(yolo, video_path, output_path=""):
    import cv2
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps = vid.get(cv2.CAP_PROP_FPS)
    video_size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False
    if isOutput:
        print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    while True:
        return_value, frame = vid.read()
        image = Image.fromarray(frame)
        image = yolo.detect_image(image)
        result = np.asarray(image)
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", result)
        if isOutput:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    yolo.close_session()

def test_time_augmentation(yolo, origin_image, origin_result, magnifiy_range=np.linspace(2, 5, 10),
                           loc_relative_error=0.2):
    '''
        Flip && Magnify i times
    '''
    for i in np.append([0], magnifiy_range):
        if i == 0:
            image_flip = origin_image.transpose(Image.FLIP_LEFT_RIGHT)
            result = yolo.detect_image(image_flip)
        else:
            image_magnified = origin_image.resize((int(origin_image.width * i), int(origin_image.height * i)))
            result = yolo.detect_image(image_magnified)

        for detect_item in result:
            top, right, bottom, left = detect_item["location"].values()
            if i == 0:
                temp = left
                left = image_flip.width - right
                right = image_flip.width - temp
            else:
                top = int((float(top) + 0.5) / float(i))
                left = int((float(left) + 0.5) / float(i))
                right = int((float(right) + 0.5) / float(i))
                bottom = int((float(bottom) + 0.5) / float(i))

            detect_item["location"]["top"] = top
            detect_item["location"]["left"] = left
            detect_item["location"]["right"] = right
            detect_item["location"]["bottom"] = bottom
            className = detect_item["class"]

            '''
                Determine whether the detected item has been recorded
            '''
            found = False
            for recorded_detect_item in origin_result:
                r_top, r_left, r_right, r_bottom = recorded_detect_item["location"].values()
                r_className = recorded_detect_item["class"]
                if className == r_className:
                    # Same Object
                    if r_top * (1 - loc_relative_error) <= top <= r_top * (1 + loc_relative_error) \
                            or r_left * (1 - loc_relative_error) <= left <= r_left * (1 + loc_relative_error) \
                            or r_right * (1 - loc_relative_error) <= right <= r_right * (1 + loc_relative_error) \
                            or r_bottom * (1 - loc_relative_error) <= bottom <= r_bottom * (1 + loc_relative_error):
                        found = True
                        break
                else:
                    # Same Object, but has different classification
                    if r_top * (1 - loc_relative_error) <= top <= r_top * (1 + loc_relative_error) \
                            and r_left * (1 - loc_relative_error) <= left <= r_left * (1 + loc_relative_error) \
                            and r_right * (1 - loc_relative_error) <= right <= r_right * (1 + loc_relative_error) \
                            and r_bottom * (1 - loc_relative_error) <= bottom <= r_bottom * (1 + loc_relative_error) \
                            and detect_item["score"] > recorded_detect_item["score"]:
                        recorded_detect_item["score"] = detect_item["score"]
                        recorded_detect_item["class"] = className
            if not found:
                origin_result.append(detect_item)

    return origin_result
