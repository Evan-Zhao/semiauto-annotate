# -*- coding: utf-8 -*-
"""
Class definition of YOLO_v3 style detection model on image and video
"""

import colorsys
from timeit import default_timer as timer

import numpy as np
import tensorflow as tf
from PIL import Image, ImageFont, ImageDraw


from yolo.yolo3.utils import load_graph, get_boxes_and_inputs_pb, letter_box_image, non_max_suppression
import os
here = os.path.abspath(os.path.dirname(__file__))


class YOLO(object):
    _defaults = {
        "model_path": os.path.join(here, 'model_data/yolo.pb'),
        "anchors_path": os.path.join(here, 'model_data/yolo_anchors.txt'),
        "classes_path": os.path.join(here, 'model_data/coco_classes.txt'),
        "score": 0.3,
        "conf_threshold":0.5,
        "iou": 0.45,
        "model_image_size": 416,
        "gpu_num": 1,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        self.__dict__.update(self._defaults)  # set up default values
        self.__dict__.update(kwargs)  # and update with user overrides
        self.image = True
        self.class_names = self._get_class()
        self.frozenGraph = load_graph(self.model_path)
        self.boxes, self.inputs = get_boxes_and_inputs_pb(self.frozenGraph)

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names


    def detect_image(self, image):
        img_resized = letter_box_image(image, self.model_image_size, self.model_image_size, 128)
        img_resized = img_resized.astype(np.float32)
        with tf.Session(graph=self.frozenGraph) as sess:
            detected_boxes = sess.run(
                self.boxes, feed_dict={self.inputs: [img_resized]})
        filtered_boxes = non_max_suppression(detected_boxes,
                                             confidence_threshold=self.conf_threshold,
                                             iou_threshold=self.iou)
        # print('Found {} boxes for {}'.format(len(out_boxes), 'img'))
        result = []
        for cls in filtered_boxes.items():
            class_name = self.class_names[cls[0]]
            for box in cls[1]:
                left, top, right, bottom = box[0]
                right *= (image.size[0] / float(self.model_image_size))
                left *= (image.size[0]/float(self.model_image_size))
                top *= (image.size[1]/float(self.model_image_size))
                bottom *= (image.size[1] / float(self.model_image_size))
                top = max(0, int(np.floor(top + 0.5)))
                left = max(0, int(np.floor(left + 0.5).astype('int')))
                bottom = min(image.size[1], int(np.floor(bottom + 0.5).astype('int')))
                right = min(image.size[0], int(np.floor(right + 0.5).astype('int')))
                box_location = {"top": top, "left": left, "right": right, "bottom": bottom}
                score = float(box[1])
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
        sess.close()
        return result

    def infer_on_image(self, image_path):
        import json

        image = Image.open(image_path)
        result = self.detect_image(image)
        #result = test_time_augmentation(self, image, result)

        # self.close_session()
        return json.dumps(result)




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
