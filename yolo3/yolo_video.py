import argparse
from yolo import YOLO
from PIL import Image, ImageDraw
import numpy as np
import json


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


def detect_img(yolo, input, output):
    image = Image.open(input)
    result = yolo.detect_image(image)
    result = test_time_augmentation(yolo, image, result)

    fo = open(output, "w")
    fo.write(json.dumps(result))
    fo.close()
    yolo.close_session()
    '''
    thickness = (image.size[0] + image.size[1]) // 300
    
    for item in result:
        draw = ImageDraw.Draw(image)
        top, right, bottom, left = item["location"].values()
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

        # My kingdom for a good redistributable image drawing library.
        for i in range(thickness):
            draw.rectangle(
                [left + i, top + i, right - i, bottom - i],
                outline=yolo.colors[9])
        del draw
    image.show()
    '''


FLAGS = None

if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--model', type=str,
        help='path to model weight file, default ' + YOLO.get_defaults("model_path")
    )

    parser.add_argument(
        '--anchors', type=str,
        help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
    )

    parser.add_argument(
        '--classes', type=str,
        help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
    )

    parser.add_argument(
        '--gpu_num', type=int,
        help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num"))
    )

    parser.add_argument(
        '--image', default=False, action="store_true",
        help='Image detection mode, will ignore all positional arguments'
    )
    '''
    Command line positional arguments -- for video detection mode
    '''
    parser.add_argument(
        "--input", nargs='?', type=str, required=False, default='./path2your_video',
        help="Video input path"
    )

    parser.add_argument(
        "--output", nargs='?', type=str, default="",
        help="[Optional] Video output path"
    )

    FLAGS = parser.parse_args()

    """
        Image detection mode, disregard any remaining command line arguments
    """
    print("Image detection mode")
    detect_img(YOLO(**vars(FLAGS)), FLAGS.input, FLAGS.output)
