import argparse
import json

import numpy as np
from PIL import Image
from yolo.yolo_class import YOLO



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


def main():
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


if __name__ == '__main__':
    main()
