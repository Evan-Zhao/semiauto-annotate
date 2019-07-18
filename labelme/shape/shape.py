import copy
import math

from qtpy.QtCore import QRectF, QPointF
from qtpy.QtGui import QPainterPath, QPen, QColor

import labelme.utils
from labelme.utils import Config
from labelme.app import Application
from .bezier import BezierB

# TODO(unknown):
# - [opt] Store paths instead of creating new ones at each paint.

DEFAULT_LINE_COLOR = QColor(0, 255, 0, 128)
DEFAULT_FILL_COLOR = QColor(255, 0, 0, 128)
DEFAULT_VERTEX_FILL_COLOR = QColor(0, 255, 0, 255)
DEFAULT_HVERTEX_FILL_COLOR = QColor(255, 0, 0)


class Shape(object):
    P_SQUARE, P_ROUND = 0, 1
    MOVE_VERTEX, NEAR_VERTEX = 0, 1
    # The following class variables influence the drawing of all shape objects.
    vertex_fill_color = DEFAULT_VERTEX_FILL_COLOR
    hvertex_fill_color = DEFAULT_HVERTEX_FILL_COLOR
    def_point_type = P_ROUND
    point_size = 8
    line_width = 4.0
    def_scale = 1.0
    _highlightSettings = {
        NEAR_VERTEX: (4, P_ROUND),
        MOVE_VERTEX: (1.5, P_SQUARE),
    }
    all_types = ['polygon', 'rectangle', 'point', 'line', 'linestrip', 'circle', 'curve', 'freeform']

    def __init__(self, points, form, shape_type):
        self.points = points
        self.form = form
        self.shape_type = shape_type
        self._highlightIndex = None
        self._highlightMode = self.NEAR_VERTEX

    @property
    def label(self):
        return self.form[0][0] if self.form else None

    def is_empty(self):
        return not self.points

    @property
    def label_color(self):
        def argb_to_q_color(val):
            hexStr = hex(val)[2:]
            a, r, g, b = tuple(int(hexStr[i:i + 2], 16) for i in (0, 2, 4, 6))
            return QColor(r, g, b, a)

        label_color = Config.get('label_color', default={})
        if self.label in label_color:
            return argb_to_q_color(label_color[self.label])
        else:
            return None

    @staticmethod
    def get_scale(canvas):
        return canvas.scale if canvas else Shape.def_scale

    @staticmethod
    def getRectFromLine(pt1, pt2):
        x1, y1 = pt1.x(), pt1.y()
        x2, y2 = pt2.x(), pt2.y()
        return QRectF(x1, y1, x2 - x1, y2 - y1)

    @staticmethod
    def getCircleRectFromLine(line):
        """Computes parameters to draw with `QPainterPath::addEllipse`"""
        if len(line) != 2:
            return None
        (c, point) = line
        r = line[0] - line[1]
        d = math.sqrt(math.pow(r.x(), 2) + math.pow(r.y(), 2))
        rectangle = QRectF(c.x() - d, c.y() - d, 2 * d, 2 * d)
        return rectangle

    @staticmethod
    def paint_vertices(painter, points, scale, highlight=None, highlight_mode=None):
        path = QPainterPath()
        if highlight is not None:
            vertex_fill_color = Shape.hvertex_fill_color
        else:
            vertex_fill_color = Shape.vertex_fill_color
        for i, point in enumerate(points):
            d = Shape.point_size / scale
            shape = Shape.def_point_type
            if i == highlight:
                size, shape = Shape._highlightSettings[highlight_mode]
                d *= size
            if shape == Shape.P_SQUARE:
                path.addRect(point.x() - d / 2, point.y() - d / 2, d, d)
            elif shape == Shape.P_ROUND:
                path.addEllipse(point, d / 2.0, d / 2.0)
            else:
                assert False, 'unsupported vertex shape'
        painter.drawPath(path)
        painter.fillPath(path, vertex_fill_color)

    @staticmethod
    def get_line_path(points, shape_type):
        line_path = QPainterPath()
        if shape_type == 'rectangle' and len(points) == 2:
            rectangle = Shape.getRectFromLine(*points)
            line_path.addRect(rectangle)
        elif shape_type == 'circle' and len(points) == 2:
            rectangle = Shape.getCircleRectFromLine(points)
            line_path.addEllipse(rectangle)
        elif shape_type == 'linestrip' and points:
            line_path.moveTo(points[0])
            for p in points:
                line_path.lineTo(p)
        elif shape_type in ['curve', 'freeform'] and points:
            # Paint Bezier curve across given points.
            refined_points = BezierB(points).smooth()
            line_path.moveTo(refined_points[0])
            for p in refined_points:
                line_path.lineTo(p)
        elif points:
            line_path.moveTo(points[0])
            for p in points:
                line_path.lineTo(p)
            if shape_type == 'polygon':
                line_path.lineTo(points[0])
        return line_path

    def set_label(self, form):
        self.form = form

    def paint(self, painter, fill=False, canvas=None):
        scale = self.get_scale(canvas)
        mainwindow = Application.get_main_window()
        color = self.label_color or mainwindow.lineColor or DEFAULT_LINE_COLOR
        pen = QPen(color)
        # Try using integer sizes for smoother drawing(?)
        pen.setWidth(max(1, int(round(self.line_width / scale))))
        painter.setPen(pen)
        # Draw all vertices
        self.paint_vertices(
            painter, self.points, scale,
            self._highlightIndex, self._highlightMode
        )
        line_path = self.get_line_path(self.points, self.shape_type)
        painter.drawPath(line_path)
        if fill:
            painter.fillPath(line_path, color)

    def nearestVertex(self, point, epsilon):
        min_distance = float('inf')
        min_i = None
        for i, p in enumerate(self.points):
            dist = labelme.utils.distance(p - point)
            if dist <= epsilon and dist < min_distance:
                min_distance = dist
                min_i = i
        return min_i

    def nearestEdge(self, point, epsilon):
        min_distance = float('inf')
        post_i = None
        for i in range(len(self.points)):
            line = [self.points[i - 1], self.points[i]]
            dist = labelme.utils.distancetoline(point, line)
            if dist <= epsilon and dist < min_distance:
                min_distance = dist
                post_i = i
        return post_i

    def containsPoint(self, point):
        return self.get_line_path(self.points, self.shape_type).contains(point)

    def boundingRect(self):
        return self.get_line_path(self.points, self.shape_type).boundingRect()

    def moveBy(self, offset):
        self.points = [p + offset for p in self.points]

    def moveVertexBy(self, i, offset):
        self.points[i] = self.points[i] + offset

    def highlightVertex(self, i, action):
        self._highlightIndex = i
        self._highlightMode = action

    def highlightClear(self):
        self._highlightIndex = None

    def copy(self):
        return copy.deepcopy(self)

    def __getstate__(self):
        return dict(
            points=self.points,
            shape_type=self.shape_type,
            form=self.form
        )

    def __setstate__(self, state):
        self.__init__(state['points'], state['form'], state['shape_type'])

    def __len__(self):
        return len(self.points)

    def __getitem__(self, key):
        return self.points[key]
