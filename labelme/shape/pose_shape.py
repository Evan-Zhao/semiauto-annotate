from typing import Iterable

from qtpy import QtCore, QtGui

from .shape import Shape


class PoseShape(Shape):
    forced_label = 'person'

    class PointIndexer:
        def __init__(self, shapes):
            self._point_to_shape = {(p.x(), p.y()): s for s in shapes for p in s}
            self._index_to_shape_idx = {}
            index, shape_index_start = 0, 0
            for s in shapes:
                shape_index_start = index
                for _ in s:
                    self._index_to_shape_idx[index] = (s, shape_index_start)
                    index += 1

        def get_shape(self, p):
            return self._point_to_shape[(p.x(), p.y())]

        def get_shape_and_index(self, i):
            shape, idx_start = self._index_to_shape_idx[i]
            return shape, i - idx_start

    def __init__(self, children, annotation=None):
        all_points = [p for s in children for p in s]
        super(PoseShape, self).__init__(
            all_points, form=[
                [PoseShape.forced_label, None, None],
                [None] * 3
            ],  # dummy form for navigation of label dialog
            shape_type=PoseShape.forced_label
        )
        if issubclass(type(annotation), dict):
            self.annotation = annotation
        elif issubclass(type(annotation), Iterable):
            self.annotation = {}
            counter = 0
            for vs in annotation:
                for v in vs:
                    self.annotation[counter] = v
                    counter += 1
        else:
            self.annotation = {}
        self._shapes = children
        self._point_indexer = PoseShape.PointIndexer(children)

    @property
    def label(self):
        return PoseShape.forced_label

    @staticmethod
    def get_paint_font(scale):
        return QtGui.QFont('Helvetica', 16 / scale)

    def paint(self, painter, fill=False, canvas=None, **kwargs):
        scale = self.get_scale(canvas)
        for s in self._shapes:
            s.paint(painter, fill, self.label_color, canvas)
        dotted_pen = QtGui.QPen(self.label_color)
        dotted_pen.setWidth(max(1, int(round(2.0 / scale))))
        dotted_pen.setStyle(QtCore.Qt.DashLine)
        painter.setPen(dotted_pen)
        dotted_line_path = QtGui.QPainterPath()
        for s1, s2 in zip(self._shapes, self._shapes[1:]):
            dotted_line_path.moveTo(s1[0])
            dotted_line_path.lineTo(s2[0])
            dotted_line_path.moveTo(s1[-1])
            dotted_line_path.lineTo(s2[-1])
        painter.drawPath(dotted_line_path)
        if self.annotation:
            text_pen = QtGui.QPen(PoseShape.def_color)
            text_pen.setWidth(max(1, int(round(1.0 / scale))))
            painter.setPen(text_pen)
            text_path = QtGui.QPainterPath()
            font = self.get_paint_font(scale)
            painter.setFont(font)
            for k, v in self.annotation.items():
                point = self.points[k]
                painter.drawText(point, str(v))
            painter.drawPath(text_path)

    def highlightVertex(self, i, action):
        """Send highlight vertex index to the shape it belongs in."""
        shape, i = self._point_indexer.get_shape_and_index(i)
        shape.highlightVertex(i, action)

    def highlightClear(self):
        """Clear highlight of ALL sub-shapes."""
        for s in self._shapes:
            s.highlightClear()

    def moveBy(self, offset):
        super(PoseShape, self).moveBy(offset)
        for s in self._shapes:
            s.moveBy(offset)

    def moveVertexBy(self, i, offset):
        super(PoseShape, self).moveVertexBy(i, offset)
        shape, i = self._point_indexer.get_shape_and_index(i)
        shape.moveVertexBy(i, offset)

    def set_label(self, annotation):
        self.annotation = annotation

    def __getstate__(self):
        return dict(
            annotation=self.annotation,
            sub_shapes=self._shapes
        )

    def __setstate__(self, state):
        annotation = {int(k): v for k, v in state['annotation'].items()}
        self.__init__(state['sub_shapes'], annotation=annotation)
