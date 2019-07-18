import os.path as osp

import jsonpickle
from qtpy import QtCore

from labelme import utils
from labelme._version import __version__


class LabelFileError(Exception):
    pass


class LabelFile(object):
    suffix = '.json'

    def __init__(self, filename=None):
        jsonpickle.set_encoder_options('json', ensure_ascii=False, indent=2)
        self.main_snapshot = None
        if filename is not None:
            self.load(filename)
        self.filename = filename

    def load(self, filename):
        if not QtCore.QFile.exists(filename) or not LabelFile.is_label_file(filename):
            raise LabelFileError()
        try:
            with open(filename, 'r') as f:
                data = jsonpickle.decode(f.read())
        except Exception as e:
            raise LabelFileError(e)
        self.filename = filename
        self.main_snapshot = data['data']

    @staticmethod
    def _get_image_dimensions(imageData):
        img_arr = utils.img_b64_to_arr(imageData)
        return img_arr.shape[0], img_arr.shape[1]

    def save(self, main_snapshot, filename):
        try:
            self.filename = filename
            data = {
                '__version__': __version__,
                'data': main_snapshot
            }
            with open(filename, 'w') as f:
                f.write(jsonpickle.encode(data))
        except Exception as e:
            raise LabelFileError(e)

    @staticmethod
    def is_label_file(filename):
        return osp.splitext(filename)[1].lower() == LabelFile.suffix

    @staticmethod
    def to_label_file_path(user_filename, output_dir):
        label_file = osp.splitext(user_filename)[0] + LabelFile.suffix
        if output_dir:
            label_file = osp.join(output_dir, osp.basename(label_file))
        return label_file
