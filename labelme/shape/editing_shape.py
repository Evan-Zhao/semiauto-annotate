from qtpy.QtGui import QColor, QPen, QPainterPath
from qtpy.QtCore import Qt, QPointF

from .bezier import BezierB
from .shape import Shape, DEFAULT_LINE_COLOR, DEFAULT_FILL_COLOR

DEFAULT_LAST_LINE_COLOR = QColor(0, 0, 255)
FREEFORM_LINE_COLOR = QColor(Qt.darkGray)
FREEFORM_FILL_COLOR = QColor(Qt.lightGray)


def fit_bezier(points):
    import numpy as np
    from scipy.interpolate import splprep, BSpline
    from itertools import groupby

    if len(points) < 3:
        return None
    coords = np.array([(p.x(), p.y()) for p in points]).transpose()
    ((t, c, k), _), _, _, _ = splprep(coords, full_output=True)
    c = np.array(c).transpose()
    unique_t = [g[0] for g in groupby(t)]
    b = BSpline(t, c, k)
    return [QPointF(x, y) for x, y in b(unique_t)]


class EditingShape(Shape):
    # The following class variables influence the drawing of all shape objects.
    must_close = ['polygon', 'rectangle', 'point', 'line', 'circle']

    def __init__(self, shape_type, init_point=None):
        points = [init_point] if init_point else []
        super().__init__(points=points, form=None, shape_type=shape_type)

    @classmethod
    def undo_into_editing_point(cls, shape):
        ret = cls(shape.shape_type)
        if shape.shape_type == 'freeform':
            return None
        if shape.shape_type == 'polygon':
            ret.points = shape.points
        elif shape.points:
            ret.points = shape.points[:-1]
        else:
            return None
        return ret

    def to_immutable_point(self):
        if self.shape_type == 'freeform':
            points = fit_bezier(self.points[:-1])
        elif self.shape_type == 'polygon':
            points = self.points[:-1]
        else:
            points = self.points
        return Shape(points, self.form, self.shape_type)

    @property
    def closed(self):
        if self.shape_type == 'polygon':
            # Polygon can only be closed with more than 2 vertices and path being a loop
            return len(self.points) > 2 and self.points[0] == self.points[-1]
        elif self.shape_type == 'point':
            # Point close when 1 point is present
            return len(self.points) == 1
        elif self.shape_type in ['rectangle', 'circle', 'line']:
            # These shapes close when 2 points are present
            return len(self.points) == 2
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

    def _get_pens_and_colors(self, canvas, line_color=None, fill_color=None):
        scale = self.get_scale(canvas)
        if self.shape_type != 'freeform':
            line_color = line_color or DEFAULT_LINE_COLOR
            fill_color = fill_color or DEFAULT_FILL_COLOR
            last_line_color = DEFAULT_LAST_LINE_COLOR
        else:
            line_color = line_color or FREEFORM_LINE_COLOR
            fill_color = fill_color or FREEFORM_FILL_COLOR
            last_line_color = FREEFORM_LINE_COLOR
        line_pen = QPen(line_color)
        last_line_pen = QPen(last_line_color)
        # Try using integer sizes for smoother drawing(?)
        width = max(1, int(round(self.line_width / scale)))
        for pen in (line_pen, last_line_pen):
            pen.setWidth(width)
        return line_pen, last_line_pen, fill_color

    def add_point(self):
        assert self.points
        self.points.append(self.points[-1])

    def update_cursor(self, value):
        assert self.points
        self.points[-1] = value

    def undo_point(self):
        if self.shape_type == 'freeform':
            self.points = []
        if self.points:
            self.points.pop()
            return True
        else:
            return False

    def insert_point(self, i, point):
        self.points.insert(i, point)

    def paint(self, painter, fill=False, canvas=None):
        from labelme.app import Application

        mainwindow = Application.get_main_window()
        line_pen, last_line_pen, fill_color = self._get_pens_and_colors(
            canvas, mainwindow.lineColor, mainwindow.fillColor
        )
        scale = self.get_scale(canvas)
        # Draw all vertices
        self.paint_vertices(
            painter, self.points, scale,
            self._highlightIndex, self._highlightMode
        )
        line_path1, line_path2 = QPainterPath(), QPainterPath()
        fill_path = line_path1
        # Get path for committed and uncommitted parts
        # respectively and draw with different colors
        if self.shape_type == 'curve' and self.points:
            # Bezier curve needs to be fitted as a whole,
            # so reimplementing this part here.
            refined_points = BezierB(self.points).smooth()
            sep_idx = refined_points.index(self.points[-1])
            line_path1.moveTo(refined_points[0])
            for p in refined_points[:sep_idx]:
                line_path1.lineTo(p)
            line_path2.moveTo(refined_points[sep_idx])
            for p in refined_points[sep_idx:]:
                line_path2.lineTo(p)
        elif self.shape_type in ['circle', 'rectangle']:
            painter.setPen(line_pen)
            line_path2 = self.get_line_path(self.points, self.shape_type)
            painter.setPen(last_line_pen)
            painter.drawPath(line_path2)
            fill_path = line_path2
        elif len(self.points) >= 2:
            # Otherwise, just get 2 different line paths,
            # painting with different colors.
            # Use type == 'line' even with polygon to prevent connecting back.
            line_path1 = self.get_line_path(self.points[:-1], 'line')
            line_path2 = self.get_line_path(self.points[-2:], 'line')
        painter.setPen(line_pen)
        painter.drawPath(line_path1)
        painter.setPen(last_line_pen)
        painter.drawPath(line_path2)
        if fill:
            painter.fillPath(fill_path, fill_color)

    def __getstate__(self):
        return dict(
            points=self.points,
            shape_type=self.shape_type
        )

    def __setstate__(self, state):
        self.__init__(state['shape_type'])
        self.points = state['points']

    def __setitem__(self, key, value):
        self.points[key] = value
