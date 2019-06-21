import copy
import math

from qtpy import QtCore
from qtpy import QtGui

import labelme.utils
from labelme import PY2

# TODO(unknown):
# - [opt] Store paths instead of creating new ones at each paint.


DEFAULT_LINE_COLOR = QtGui.QColor(0, 255, 0, 128)
DEFAULT_FILL_COLOR = QtGui.QColor(255, 0, 0, 128)
DEFAULT_SELECT_LINE_COLOR = QtGui.QColor(255, 255, 255)
DEFAULT_SELECT_FILL_COLOR = QtGui.QColor(0, 128, 255, 155)
DEFAULT_VERTEX_FILL_COLOR = QtGui.QColor(0, 255, 0, 255)
DEFAULT_HVERTEX_FILL_COLOR = QtGui.QColor(255, 0, 0)


class BezierB:
    def __init__(self, points):
        self.points = points

    @staticmethod
    def mid_point(p1, p2):
        return (p1 + p2) / 2

    @staticmethod
    def window(n, seq):
        """
        Returns a sliding window (of width n) over data from the iterable
           s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...
        """
        from itertools import islice
        it = iter(seq)
        result = tuple(islice(it, n))
        if len(result) == n:
            yield result
        for elem in it:
            result = result[1:] + (elem,)
            yield result

    @staticmethod
    def tuck(p0, pl, pr, s):
        laplace = ((pl - p0) + (pr - p0)) / 2
        return p0 + s * laplace

    def refine(self):
        if len(self.points) < 2:
            return
        points = [self.points[0]]
        for p0, p1 in self.window(2, self.points):
            points.extend([self.mid_point(p0, p1), p1])
        self.points = points

    def tuck_all(self, s):
        if len(self.points) < 3:
            return
        points = [self.points[0]]
        for pl, p0, pr in self.window(3, self.points):
            points.append(self.tuck(p0, pl, pr, s))
        points.append(self.points[-1])
        self.points = points

    def smooth(self, smoothness=5):
        for i in range(smoothness):
            self.refine()
            self.tuck_all(1 / 2)
            self.tuck_all(-1)
        return self.points


class Shape(object):
    P_SQUARE, P_ROUND = 0, 1

    MOVE_VERTEX, NEAR_VERTEX = 0, 1

    # The following class variables influence the drawing of all shape objects.
    line_color = DEFAULT_LINE_COLOR
    fill_color = DEFAULT_FILL_COLOR
    select_line_color = DEFAULT_SELECT_LINE_COLOR
    select_fill_color = DEFAULT_SELECT_FILL_COLOR
    vertex_fill_color = DEFAULT_VERTEX_FILL_COLOR
    hvertex_fill_color = DEFAULT_HVERTEX_FILL_COLOR
    point_type = P_ROUND
    point_size = 8
    scale = 1.0

    def __init__(self, label=None, line_color=None, shape_type=None):
        self.label = label
        self.points = []
        self.fill = False
        self.selected = False
        self.shape_type = shape_type
        self.form = None

        self._highlightIndex = None
        self._highlightMode = self.NEAR_VERTEX
        self._highlightSettings = {
            self.NEAR_VERTEX: (4, self.P_ROUND),
            self.MOVE_VERTEX: (1.5, self.P_SQUARE),
        }

        self._closed = False

        if line_color is not None:
            # Override the class line_color attribute
            # with an object attribute. Currently this
            # is used for drawing the pending line a different color.
            self.line_color = line_color

        self.shape_type = shape_type

    @property
    def shape_type(self):
        return self._shape_type

    @shape_type.setter
    def shape_type(self, value):
        if value is None:
            value = 'polygon'
        if value not in ['polygon', 'rectangle', 'point',
                         'line', 'circle', 'linestrip', 'curve', 'freeform']:
            raise ValueError('Unexpected shape_type: {}'.format(value))
        self._shape_type = value

    def close(self):
        self._closed = True

    def addPoint(self, point):
        if self.points and point == self.points[0]:
            self.close()
        else:
            self.points.append(point)

    def popPoint(self):
        if self.points:
            return self.points.pop()
        return None

    def insertPoint(self, i, point):
        self.points.insert(i, point)

    def isClosed(self):
        return self._closed

    def setOpen(self):
        self._closed = False

    def getRectFromLine(self, pt1, pt2):
        x1, y1 = pt1.x(), pt1.y()
        x2, y2 = pt2.x(), pt2.y()
        return QtCore.QRectF(x1, y1, x2 - x1, y2 - y1)

    def paint(self, painter):
        if not self.points:
            return
        color = self.line_color
        pen = QtGui.QPen(color)
        # Try using integer sizes for smoother drawing(?)
        pen.setWidth(max(1, int(round(2.0 / self.scale))))
        painter.setPen(pen)

        line_path = QtGui.QPainterPath()
        vrtx_path = QtGui.QPainterPath()

        if self.shape_type == 'rectangle':
            assert len(self.points) in [1, 2]
            if len(self.points) == 2:
                rectangle = self.getRectFromLine(*self.points)
                line_path.addRect(rectangle)
            for i in range(len(self.points)):
                self.drawVertex(vrtx_path, i)
        elif self.shape_type == "circle":
            assert len(self.points) in [1, 2]
            if len(self.points) == 2:
                rectangle = self.getCircleRectFromLine(self.points)
                line_path.addEllipse(rectangle)
            for i in range(len(self.points)):
                self.drawVertex(vrtx_path, i)
        elif self.shape_type == "linestrip":
            line_path.moveTo(self.points[0])
            for i, p in enumerate(self.points):
                line_path.lineTo(p)
                self.drawVertex(vrtx_path, i)
        elif self.shape_type == 'curve':
            for i in range(len(self.points)):
                self.drawVertex(vrtx_path, i)
            # Paint Bezier curve across given points.
            refined_points = BezierB(self.points).smooth()
            line_path.moveTo(self.points[0])
            for p in refined_points:
                line_path.lineTo(p)
        else:
            line_path.moveTo(self.points[0])
            # Uncommenting the following line will draw 2 paths
            # for the 1st vertex, and make it non-filled, which
            # may be desirable.
            # self.drawVertex(vrtx_path, 0)

            for i, p in enumerate(self.points):
                line_path.lineTo(p)
                self.drawVertex(vrtx_path, i)
            if self.isClosed():
                line_path.lineTo(self.points[0])

        painter.drawPath(line_path)
        painter.drawPath(vrtx_path)
        painter.fillPath(vrtx_path, self.vertex_fill_color)
        if self.fill:
            color = self.fill_color
            painter.fillPath(line_path, color)

    def drawVertex(self, path, i):
        d = self.point_size / self.scale
        shape = self.point_type
        point = self.points[i]
        if i == self._highlightIndex:
            size, shape = self._highlightSettings[self._highlightMode]
            d *= size
        if self._highlightIndex is not None:
            self.vertex_fill_color = self.hvertex_fill_color
        else:
            self.vertex_fill_color = Shape.vertex_fill_color
        if shape == self.P_SQUARE:
            path.addRect(point.x() - d / 2, point.y() - d / 2, d, d)
        elif shape == self.P_ROUND:
            path.addEllipse(point, d / 2.0, d / 2.0)
        else:
            assert False, "unsupported vertex shape"

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
        return self.makePath().contains(point)

    def getCircleRectFromLine(self, line):
        """Computes parameters to draw with `QPainterPath::addEllipse`"""
        if len(line) != 2:
            return None
        (c, point) = line
        r = line[0] - line[1]
        d = math.sqrt(math.pow(r.x(), 2) + math.pow(r.y(), 2))
        rectangle = QtCore.QRectF(c.x() - d, c.y() - d, 2 * d, 2 * d)
        return rectangle

    def makePath(self):
        if self.shape_type == 'rectangle':
            path = QtGui.QPainterPath()
            if len(self.points) == 2:
                rectangle = self.getRectFromLine(*self.points)
                path.addRect(rectangle)
        elif self.shape_type == "circle":
            path = QtGui.QPainterPath()
            if len(self.points) == 2:
                rectangle = self.getCircleRectFromLine(self.points)
                path.addEllipse(rectangle)
        else:
            path = QtGui.QPainterPath(self.points[0])
            for p in self.points[1:]:
                path.lineTo(p)
        return path

    def boundingRect(self):
        return self.makePath().boundingRect()

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

    def toJson(self, defLineColor, defFillColor):
        return dict(
            label=self.label.encode('utf-8') if PY2 else self.label,
            line_color=self.line_color.getRgb()
            if self.line_color != defLineColor else None,
            fill_color=self.fill_color.getRgb()
            if self.fill_color != defFillColor else None,
            points=[(p.x(), p.y()) for p in self.points],
            shape_type=self.shape_type,
            form=self.form
        )

    def __len__(self):
        return len(self.points)

    def __getitem__(self, key):
        return self.points[key]

    def __setitem__(self, key, value):
        self.points[key] = value
