from qtpy.QtGui import QColor, QFont, QPen, QPainterPath

from labelme.app import Application
from .shape import Shape, DEFAULT_LINE_COLOR

TEXT_COLOR = QColor(0, 255, 0, 128)


class PoseShape(Shape):
    forced_type = 'linestrip'
    forced_label = 'person'
    body_segs = [
        [1, 2, 3, 4], [1, 5, 6, 7],
        [1, 8, 9, 10], [1, 11, 12, 13], [1, 0],
        [0, 14, 16, 2], [0, 15, 17, 5]
    ]
    n_pose_points = 18

    def __init__(self, maybe_points):
        points = [p for p in maybe_points if p is not None]
        super(PoseShape, self).__init__(
            points, form=[
                [PoseShape.forced_label, None, None]
            ],  # dummy form for navigation of label dialog
            shape_type=PoseShape.forced_type
        )
        # self.maybe_points = maybe_points
        self.point_to_label = self.make_point_to_label(maybe_points)

    @staticmethod
    def make_point_to_label(maybe_points):
        shape_idx_to_body_idx = {}
        real_idx = 0
        for i, p in enumerate(maybe_points):
            if p is None:
                continue
            shape_idx_to_body_idx[real_idx] = i
            real_idx += 1
        return shape_idx_to_body_idx

    @property
    def label(self):
        return PoseShape.forced_label

    @property
    def maybe_points(self):
        ret = [None for _ in range(self.n_pose_points)]
        for i1, i2 in self.point_to_label.items():
            ret[i2] = self.points[i1]
        return ret

    @property
    def chains(self):
        chains = []
        for seg in self.body_segs:
            points = [p for p in [self.maybe_points[i] for i in seg] if p is not None]
            if not points:
                continue
            chains.append(points)
        return chains

    @staticmethod
    def get_paint_font(scale):
        return QFont('Helvetica', 16 / scale)

    def paint_chain(self, painter, chain, color, scale):
        pen = QPen(color)
        # Try using integer sizes for smoother drawing(?)
        pen.setWidth(max(1, int(round(self.line_width / scale))))
        painter.setPen(pen)
        line_path = self.get_line_path(chain, self.shape_type)
        painter.drawPath(line_path)

    def paint(self, painter, fill=False, canvas=None):
        scale = self.get_scale(canvas)
        # Draw all vertices
        self.paint_vertices(
            painter, self.points, scale,
            self._highlightIndex, self._highlightMode
        )
        mainwindow = Application.get_main_window()
        color = self.label_color or mainwindow.lineColor or DEFAULT_LINE_COLOR
        for ch in self.chains:
            self.paint_chain(painter, ch, color, scale)
        for i, p in enumerate(self.points):
            text_pen = QPen(TEXT_COLOR)
            text_pen.setWidth(max(1, int(round(1.0 / scale))))
            painter.setPen(text_pen)
            text_path = QPainterPath()
            font = self.get_paint_font(scale)
            painter.setFont(font)
            label = str(self.point_to_label[i])
            painter.drawText(p, label)
            painter.drawPath(text_path)
        if fill:
            fill_path = self.get_line_path(self.points, 'polygon')
            painter.fillPath(fill_path, color)

    def __getstate__(self):
        return dict(
            maybe_points=self.maybe_points
        )

    def __setstate__(self, state):
        self.__init__(state['maybe_points'])
