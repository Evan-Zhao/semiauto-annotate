import base64
import io

import numpy as np
import PIL.ExifTags
import PIL.Image
import PIL.ImageOps

from .config import Config


def img_b64_to_arr(img_b64):
    f = io.BytesIO()
    f.write(base64.b64decode(img_b64))
    img_arr = np.array(PIL.Image.open(f))
    return img_arr


def img_arr_to_b64(img_arr):
    img_pil = PIL.Image.fromarray(img_arr)
    f = io.BytesIO()
    img_pil.save(f, format='PNG')
    img_bin = f.getvalue()
    if hasattr(base64, 'encodebytes'):
        img_b64 = base64.encodebytes(img_bin)
    else:
        img_b64 = base64.encodestring(img_bin)
    return img_b64


def apply_exif_orientation(image):
    try:
        exif = image._getexif()
    except AttributeError:
        exif = None

    if exif is None:
        return image

    exif = {
        PIL.ExifTags.TAGS[k]: v
        for k, v in exif.items()
        if k in PIL.ExifTags.TAGS
    }

    orientation = exif.get('Orientation', None)

    if orientation == 1:
        # do nothing
        return image
    elif orientation == 2:
        # left-to-right mirror
        return PIL.ImageOps.mirror(image)
    elif orientation == 3:
        # rotate 180
        return image.transpose(PIL.Image.ROTATE_180)
    elif orientation == 4:
        # top-to-bottom mirror
        return PIL.ImageOps.flip(image)
    elif orientation == 5:
        # top-to-left mirror
        return PIL.ImageOps.mirror(image.transpose(PIL.Image.ROTATE_270))
    elif orientation == 6:
        # rotate 270
        return image.transpose(PIL.Image.ROTATE_270)
    elif orientation == 7:
        # top-to-right mirror
        return PIL.ImageOps.mirror(image.transpose(PIL.Image.ROTATE_90))
    elif orientation == 8:
        # rotate 90
        return image.transpose(PIL.Image.ROTATE_90)
    else:
        return image


class ImageFileIOError(Exception):
    pass


class ImageUnsupportedError(Exception):
    pass


class ImageFile(object):
    def __init__(self, filename=None):
        self.data = None
        self.image = None
        self.pixmap = None
        if filename is not None:
            self.data = self._read_data(filename)
            self.image = self._to_image(self.data)
            self.pixmap = self._to_pixmap(self.image)
        self.filename = filename

    @staticmethod
    def _read_data(filename):
        import os.path as osp

        try:
            image_pil = PIL.Image.open(filename)
        except IOError as e:
            raise ImageFileIOError(e)

        # apply orientation to image according to exif
        image_pil = apply_exif_orientation(image_pil)

        with io.BytesIO() as f:
            ext = osp.splitext(filename)[1].lower()
            if ext in ['.jpg', '.jpeg']:
                extension = 'JPEG'
            else:
                extension = 'PNG'
            image_pil.save(f, format=extension)
            f.seek(0)
            return f.read()

    @staticmethod
    def _to_image(image_data):
        from qtpy import QtGui
        image = QtGui.QImage.fromData(image_data)
        if image.isNull():
            raise ImageUnsupportedError
        return image

    @staticmethod
    def _to_pixmap(image):
        from qtpy import QtGui
        return QtGui.QPixmap.fromImage(image)

    @staticmethod
    def encode_data(data):
        return base64.b64encode(data).decode('utf-8')

    def __getstate__(self):
        # osp.relpath(['imagePath'], osp.dirname(filename))
        return {
            'image_data': self.encode_data(self.data) if Config.get('store_data') else None,
            'image_path': self.filename
        }

    def __setstate__(self, state):
        self.__init__()
        # osp.join(osp.dirname(filename), data['imagePath'])
        if state['image_data'] is not None:
            self.data = base64.b64decode(state['image_data'])
        else:
            self.data = self._read_data(state['image_path'])
        self.image = self._to_image(self.data)
        self.pixmap = self._to_pixmap(self.image)
        self.filename = state['image_path']
