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
DEFAULT_LAST_LINE_COLOR = QtGui.QColor(0, 0, 255)


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

    all_types = ['polygon', 'rectangle', 'point', 'line', 'circle', 'linestrip', 'curve', 'freeform']
    must_close = ['polygon', 'rectangle', 'point', 'line', 'circle']
    manual_close = ['polygon', 'curve', 'freeform']

    def __init__(self, label=None, line_color=None, shape_type=None, init_point=None):
        self.label = label
        self.form = None
        self._points = []
        self._cursor_point = None
        if init_point:
            self._points.append(init_point)
            self._cursor_point = init_point
        self.fill = False
        self.shape_type = shape_type

        self._highlightIndex = None
        self._highlightMode = self.NEAR_VERTEX
        self._highlightSettings = {
            self.NEAR_VERTEX: (4, self.P_ROUND),
            self.MOVE_VERTEX: (1.5, self.P_SQUARE),
        }

        self._polygon_closed = False
        self._immutable = False

        if line_color is not None:
            # Override the class line_color attribute
            # with an object attribute. Currently this
            # is used for drawing the pending line a different color.
            self.line_color = line_color

        self.cursor_line_color = self.line_color
        self.shape_type = shape_type

    @classmethod
    def from_list(cls, lst, **kwargs):
        inst = cls(**kwargs)
        inst._points = lst
        inst._immutable = True
        return inst

    @property
    def shape_type(self):
        return self._shape_type

    @shape_type.setter
    def shape_type(self, value):
        if value is None:
            value = 'polygon'
        if value not in self.all_types:
            raise ValueError('Unexpected shape_type: {}'.format(value))
        self._shape_type = value

    @property
    def closed(self):
        if self.shape_type in self.manual_close:
            # Polygon can only be closed if path is a loop
            return self._polygon_closed
        elif self.shape_type == 'point':
            # Point close when 1 point is present
            return len(self._points) == 1
        elif self.shape_type in ['rectangle', 'circle', 'line']:
            # These shapes close when 2 points are present
            return len(self._points) == 2
        else:
            return False

    @property
    def complete(self):
        # Polygon/point/rect/circle/line must be closed to be a complete shape
        # Other shapes are complete at any time
        if self.shape_type in self.must_close:
            return self.closed
        else:
            return True

    @property
    def last_line_color(self):
        return self.line_color if self.closed else DEFAULT_LAST_LINE_COLOR

    def set_immutable(self):
        self._immutable = True
        self._cursor_point = None

    def commit_point(self, cursor_point=None):
        assert self._cursor_point is not None
        if self.shape_type == 'polygon' and self._points[0] == self._cursor_point:
            self._polygon_closed = True
        else:
            self._points.append(self._cursor_point)
        self._cursor_point = cursor_point

    def update_cursor(self, value):
        assert self._cursor_point is not None
        self._cursor_point = value

    def undo_point(self, forced=False):
        if self._polygon_closed:
            if forced:
                self._polygon_closed = False
                return True
            else:
                return False
        if self._points:
            self._cursor_point = self._points.pop()
            return True
        else:
            return False

    def insertPoint(self, i, point):
        self._points.insert(i, point)

    def is_empty(self):
        return not self._points

    @staticmethod
    def getRectFromLine(pt1, pt2):
        x1, y1 = pt1.x(), pt1.y()
        x2, y2 = pt2.x(), pt2.y()
        return QtCore.QRectF(x1, y1, x2 - x1, y2 - y1)

    def paint(self, painter):
        actual_points = (self._points + [self._cursor_point]) if self._cursor_point else self._points

        def drawVertex(path, idx):
            d = self.point_size / self.scale
            shape = self.point_type
            point = actual_points[idx]
            if idx == self._highlightIndex:
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
                assert False, 'unsupported vertex shape'

        def draw_vertices(painter):
            vrtx_path = QtGui.QPainterPath()
            for i in range(len(actual_points)):
                drawVertex(vrtx_path, i)
            painter.drawPath(vrtx_path)
            painter.fillPath(vrtx_path, self.vertex_fill_color)

        pen1, pen2 = QtGui.QPen(self.line_color), QtGui.QPen(self.last_line_color)
        # Try using integer sizes for smoother drawing(?)
        pen1.setWidth(max(1, int(round(2.0 / self.scale))))
        painter.setPen(pen1)
        draw_vertices(painter)

        line_path1, line_path2 = QtGui.QPainterPath(), QtGui.QPainterPath()
        single_line_path = line_path2 if self._cursor_point else line_path1
        if self.shape_type == 'rectangle' and len(actual_points) == 2:
            rectangle = self.getRectFromLine(*actual_points)
            single_line_path.addRect(rectangle)
        elif self.shape_type == 'circle' and len(actual_points) == 2:
            rectangle = self.getCircleRectFromLine(actual_points)
            single_line_path.addEllipse(rectangle)
        elif self.shape_type == 'linestrip' and self._points:
            line_path1.moveTo(self._points[0])
            for p in self._points:
                line_path1.lineTo(p)
            if self._cursor_point:
                line_path2.moveTo(self._points[-1])
                line_path2.lineTo(self._cursor_point)
        elif self.shape_type == 'curve' and self._points:
            # Paint Bezier curve across given points.
            refined_points = BezierB(actual_points).smooth()
            if self._cursor_point:
                sep_idx = refined_points.index(self._points[-1])
                line_path1.moveTo(refined_points[0])
                for p in refined_points[:sep_idx]:
                    line_path1.lineTo(p)
                line_path2.moveTo(refined_points[sep_idx])
                for p in refined_points[sep_idx:]:
                    line_path2.lineTo(p)
            else:
                line_path1.moveTo(refined_points[0])
                for p in refined_points:
                    line_path1.lineTo(p)
        elif self._points:
            line_path1.moveTo(actual_points[0])
            for p in actual_points:
                line_path1.lineTo(p)
            if self._cursor_point:
                line_path2.moveTo(self._points[-1])
                line_path2.lineTo(self._cursor_point)
            elif self.closed:
                line_path1.lineTo(actual_points[0])

        painter.drawPath(line_path1)
        painter.setPen(pen2)
        painter.drawPath(line_path2)
        if self.fill:
            color = self.fill_color
            painter.fillPath(line_path1, color)

    def nearestVertex(self, point, epsilon):
        min_distance = float('inf')
        min_i = None
        for i, p in enumerate(self._points):
            dist = labelme.utils.distance(p - point)
            if dist <= epsilon and dist < min_distance:
                min_distance = dist
                min_i = i
        return min_i

    def nearestEdge(self, point, epsilon):
        min_distance = float('inf')
        post_i = None
        for i in range(len(self._points)):
            line = [self._points[i - 1], self._points[i]]
            dist = labelme.utils.distancetoline(point, line)
            if dist <= epsilon and dist < min_distance:
                min_distance = dist
                post_i = i
        return post_i

    def containsPoint(self, point):
        return self.makePath().contains(point)

    @staticmethod
    def getCircleRectFromLine(line):
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
            if len(self._points) == 2:
                rectangle = self.getRectFromLine(*self._points)
                path.addRect(rectangle)
        elif self.shape_type == 'circle':
            path = QtGui.QPainterPath()
            if len(self._points) == 2:
                rectangle = self.getCircleRectFromLine(self._points)
                path.addEllipse(rectangle)
        else:
            path = QtGui.QPainterPath(self._points[0])
            for p in self._points[1:]:
                path.lineTo(p)
        return path

    def boundingRect(self):
        return self.makePath().boundingRect()

    def moveBy(self, offset):
        self._points = [p + offset for p in self._points]

    def moveVertexBy(self, i, offset):
        self._points[i] = self._points[i] + offset

    def highlightVertex(self, i, action):
        self._highlightIndex = i
        self._highlightMode = action

    def highlightClear(self):
        self._highlightIndex = None

    def copy(self):
        return copy.deepcopy(self)

    def __getstate__(self):
        return dict(
            label=self.label.encode('utf-8') if PY2 else self.label,
            line_color=self.line_color.getRgb(),
            fill_color=self.fill_color.getRgb(),
            points=[(p.x(), p.y()) for p in self._points],
            shape_type=self.shape_type,
            form=self.form
        )

    def __setstate__(self, state):
        self.__init__()
        self.label = state['label']
        self.line_color = QtGui.QColor(*state['line_color'])
        self.fill_color = QtGui.QColor(*state['fill_color'])
        self._points = [QtCore.QPointF(*p) for p in state['points']]
        self.shape_type = state['shape_type']
        self.form = state['form']

    def __len__(self):
        return len(self._points)

    def __getitem__(self, key):
        return self._points[key]

    def __setitem__(self, key, value):
        self._points[key] = value


class MultiShape(Shape):
    def __init__(self, shapes):
        """
        Initialize super class with all points of all shapes
        to reuse some of the Shapes function, like nearestVertex
        """
        super().__init__()
        all_points = [p for s in shapes for p in s]
        self._points = all_points
        self._shapes = shapes
        self.label = shapes[0].label if shapes else None
        self.line_color = shapes[0].line_color if shapes else None
        self.fill_color = self.line_color
        self.form = None

    def paint(self, painter):
        for s in self._shapes:
            s.paint(painter)
        pen = QtGui.QPen(self.line_color)
        # Wider line than shape itself
        pen.setWidth(max(1, int(round(4.0 / self.scale))))
        pen.setStyle(QtCore.Qt.DashLine)
        painter.setPen(pen)
        path = QtGui.QPainterPath()
        for s1, s2 in zip(self._shapes, self._shapes[1:]):
            path.moveTo(s1[0])
            path.lineTo(s2[0])
            path.moveTo(s1[-1])
            path.lineTo(s2[-1])
        painter.drawPath(path)

    def __getstate__(self):
        return dict(
            label=self.label.encode('utf-8') if PY2 else self.label,
            line_color=self.line_color,
            fill_color=self.fill_color,
            sub_shapes=self._shapes,
            form=self.form
        )

    def __setstate__(self, state):
        self.__init__()
        self.label = state['label']
        self.line_color = state['line_color']
        self.fill_color = state['fill_color']
        self._shapes = state['sub_shapes']
        self.form = state['form']
