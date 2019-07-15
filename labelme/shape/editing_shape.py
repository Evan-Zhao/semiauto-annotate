from qtpy import QtGui

from .bezier import BezierB
from .shape import Shape

DEFAULT_LAST_LINE_COLOR = QtGui.QColor(0, 0, 255)
DEFAULT_LINE_COLOR = QtGui.QColor(0, 255, 0, 128)


class EditingShape(Shape):
    # The following class variables influence the drawing of all shape objects.
    def_color = DEFAULT_LINE_COLOR
    def_last_color = DEFAULT_LAST_LINE_COLOR
    must_close = ['polygon', 'rectangle', 'point', 'line', 'circle']
    manual_close = ['polygon', 'curve', 'freeform']

    def __init__(self, shape_type, init_point=None):
        points = [init_point] if init_point else []
        super().__init__(points=points, form=None, shape_type=shape_type)

    # TODO
    @classmethod
    def undo_into_editing_point(cls, shape):
        ret = cls(shape.shape_type)
        if shape.shape_type == 'polygon':
            ret.points = shape.points
        elif shape.points:
            ret.points = shape.points[:-1]
            # add ret.points[-1]
        else:
            return None
        return ret

    def to_immutable_point(self):
        points = self.points[:-1] if self.shape_type == 'polygon' else self.points
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

    def add_point(self):
        assert self.points
        self.points.append(self.points[-1])

    def update_cursor(self, value):
        assert self.points
        self.points[-1] = value

    def undo_point(self):
        if self.points:
            self.points.pop()
            return True
        else:
            return False

    def insert_point(self, i, point):
        self.points.insert(i, point)

    def paint(self, painter, fill=False, canvas=None, **kwargs):
        scale = self.get_scale(canvas)
        pen1 = QtGui.QPen(EditingShape.def_color)
        pen2 = QtGui.QPen(EditingShape.def_last_color)
        # Try using integer sizes for smoother drawing(?)
        width = max(1, int(round(self.line_width / scale)))
        pen1.setWidth(width)
        pen2.setWidth(width)
        painter.setPen(pen1)
        # Draw all vertices
        self.paint_vertices(
            painter, self.points, scale,
            self._highlightIndex, self._highlightMode
        )
        # Get path for committed and uncommitted parts
        # respectively and draw with different colors
        if self.shape_type == 'curve' and self.points:
            # Bezier curve needs to be fitted as a whole,
            # so reimplementing this part here.
            line_path1, line_path2 = QtGui.QPainterPath(), QtGui.QPainterPath()
            refined_points = BezierB(self.points).smooth()
            sep_idx = refined_points.index(self.points[-1])
            line_path1.moveTo(refined_points[0])
            for p in refined_points[:sep_idx]:
                line_path1.lineTo(p)
            line_path2.moveTo(refined_points[sep_idx])
            for p in refined_points[sep_idx:]:
                line_path2.lineTo(p)
            painter.drawPath(line_path1)
            painter.setPen(pen2)
            painter.drawPath(line_path2)
            if fill:
                painter.fillPath(line_path1, EditingShape.def_color)
        elif self.shape_type in ['circle', 'rectangle']:
            line_path2 = self.get_line_path(self.points, self.shape_type)
            painter.setPen(pen2)
            painter.drawPath(line_path2)
        elif len(self.points) >= 2:
            # Otherwise, just get 2 different line paths,
            # painting with different colors.
            # Use type == 'line' even with polygon to prevent connecting back.
            line_path1 = self.get_line_path(self.points[:-1], 'line')
            line_path2 = self.get_line_path(self.points[-2:], 'line')
            painter.drawPath(line_path1)
            painter.setPen(pen2)
            painter.drawPath(line_path2)
            if fill:
                painter.fillPath(line_path1, EditingShape.def_color)

    def __getstate__(self):
        raise RuntimeError('Editing shape should not be pickled.')

    def __setstate__(self, state):
        raise RuntimeError('Editing shape should not be unpickled.')

    def __setitem__(self, key, value):
        self.points[key] = value
