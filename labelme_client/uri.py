import os
import os.path as osp

from qtpy.QtCore import QObject
from qtpy.QtGui import QImageReader


def scan_all_images(folder_path):
    extensions = ['.%s' % fmt.data().decode("ascii").lower()
                  for fmt in QImageReader.supportedImageFormats()]
    images = []

    for root, _, files in os.walk(folder_path):
        root_to_folder = os.path.relpath(root, folder_path)
        for file in files:
            if file.lower().endswith(tuple(extensions)):
                relativePath = osp.normpath(osp.join(root_to_folder, file))
                images.append(relativePath)
    images.sort(key=lambda x: x.lower())
    return images


def find_dir_images(dir_path, label_file_dir):
    ret = []
    for image_file in scan_all_images(dir_path):
        label_filename = osp.splitext(image_file)[0] + '.json'
        label_file_abspath = osp.join(label_file_dir, label_filename)
        label_file_relpath = osp.relpath(label_file_abspath, dir_path)
        if os.path.isfile(label_file_abspath):
            ret.append((label_file_relpath, True))
        else:
            ret.append((image_file, False))
    return ret


class URI(QObject):
    def __init__(self, is_file_like, name, file_list):
        super().__init__()
        self.is_file_like = is_file_like
        self.name = name
        self.file_list = file_list
        self.idx_of_list = 0

    @classmethod
    def from_server(cls, server):
        raise NotImplementedError()

    @classmethod
    def from_folder(cls, dir_path, label_file_dir):
        if label_file_dir is None:
            label_file_dir = dir_path
        return cls(False, dir_path, find_dir_images(dir_path, label_file_dir))

    @classmethod
    def from_file(cls, file_name):
        return cls(True, file_name, [])

    @classmethod
    def from_file_or_folder(cls, file_or_folder, output_dir):
        if not osp.exists(file_or_folder) or osp.isfile(file_or_folder):
            return URI.from_file(file_or_folder)
        else:
            return URI.from_folder(file_or_folder, output_dir)

    @property
    def filename(self):
        return self[self.idx_of_list]

    @property
    def folder_name(self):
        if self.is_file_like:
            return osp.dirname(self.name)
        else:
            return self.name

    def _shift(self, n):
        if not self.file_list:
            return None
        ln = len(self.file_list)
        return (self.idx_of_list + n) % ln

    def get_next_idx(self):
        return self._shift(1)

    def get_prev_idx(self):
        return self._shift(-1)

    def set_selection(self, n):
        self.idx_of_list = n

    def filter(self, pattern):
        self.file_list = [
            (path, b) for path, b in self.file_list
            if pattern in path
        ]

    def set_current_file_saved(self, saved_to_abs):
        self.file_list[self.idx_of_list] = \
            osp.relpath(saved_to_abs, self.name), True

    def output_dir_changed(self, output_dir):
        self.file_list = find_dir_images(self.name, output_dir)

    def __getitem__(self, item):
        if self.is_file_like:
            return self.name
        elif not self.file_list:
            return None
        else:
            return osp.normpath(
                osp.join(self.name, self.file_list[item][0])
            )
