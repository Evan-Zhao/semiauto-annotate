from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets

import labelme.utils
from labelme import QT5
from labelme.shape import Shape, MultiShape
from labelme.custom_widgets.join_shapes_dialog import JoinShapesDialog
from labelme.utils import Config

# TODO(unknown):
# - [maybe] Find optimal epsilon value.


CURSOR_DEFAULT = QtCore.Qt.ArrowCursor
CURSOR_POINT = QtCore.Qt.PointingHandCursor
CURSOR_DRAW = QtCore.Qt.CrossCursor
CURSOR_MOVE = QtCore.Qt.ClosedHandCursor
CURSOR_GRAB = QtCore.Qt.OpenHandCursor


class Canvas(QtWidgets.QWidget):
    zoomRequest = QtCore.Signal(int, QtCore.QPointF)
    scrollRequest = QtCore.Signal(int, int)
    newShape = QtCore.Signal()
    mergeShape = QtCore.Signal(list, Shape)
    selectionChanged = QtCore.Signal()
    shapeMoved = QtCore.Signal()
    drawingPolygon = QtCore.Signal(bool)
    edgeSelected = QtCore.Signal(bool)

    CREATE, EDIT = 0, 1

    # polygon, rectangle, line, or point
    _createMode = 'polygon'
    _fill_drawing = False

    def __init__(self, *args, **kwargs):
        # Configs
        self._epsilon = Config.get('epsilon', default=10.0)
        self._label_color = Config.get('label_color')
        super(Canvas, self).__init__(*args, **kwargs)
        # Initialise local state.
        self._image_file = None
        self._mode = self.EDIT
        self._current = None
        # Previous points don't exist at init time,
        # but image position exists and is (0, 0).
        self._prevPoint, self._prevMovePoint = None, None
        self._offsets = None, None
        self._imagePos, self.scale = QtCore.QPointF(), 1.0
        self._hideBackground = False
        self.hideBackground = False
        self._hShape, self._hVertex, self._hEdge = None, None, None
        self._movingShape = False
        self._painter = QtGui.QPainter()
        self._cursor = CURSOR_DEFAULT
        # Public state
        self.shapes = []
        self.shapesBackups = []
        self.selectedShapes = []
        self.selectedShapesCopy = []
        self.pixmap = None
        self.visible = {}
        # Menus:
        # 0: right-click without selection and dragging of shapes
        # 1: right-click with selection and dragging of shapes
        self.menus = (QtWidgets.QMenu(), QtWidgets.QMenu())
        # Set widget options.
        self.setMouseTracking(True)
        self.setFocusPolicy(QtCore.Qt.WheelFocus)

    def fillDrawing(self):
        return self._fill_drawing

    def setFillDrawing(self, value):
        self._fill_drawing = value

    @property
    def createMode(self):
        return self._createMode

    @createMode.setter
    def createMode(self, value):
        if value not in ['polygon', 'rectangle', 'circle',
                         'line', 'point', 'linestrip', 'curve', 'freeform']:
            raise ValueError('Unsupported createMode: %s' % value)
        self._createMode = value

    @property
    def snapshot(self):
        return {
            'shapes': self.shapes,
            'image_file': self._image_file
        }

    def load_snapshot(self, value):
        # Loading image clears all shapes, so it goes first.
        self.load_image_file(value['image_file'])
        self.loadShapes(value['shapes'])

    def is_empty(self):
        return self.pixmap is None

    def storeShapes(self):
        shapesBackup = []
        for shape in self.shapes:
            shapesBackup.append(shape.copy())
        if len(self.shapesBackups) >= 10:
            self.shapesBackups = self.shapesBackups[-9:]
        self.shapesBackups.append(shapesBackup)

    @property
    def isShapeRestorable(self):
        if len(self.shapesBackups) < 2:
            return False
        return True

    def restoreShape(self):
        if not self.isShapeRestorable:
            return
        self.shapesBackups.pop()  # latest
        shapesBackup = self.shapesBackups.pop()
        self.shapes = shapesBackup
        self.selectedShapes = []
        self.repaint()

    def enterEvent(self, ev):
        self.overrideCursor(self._cursor)

    def leaveEvent(self, ev):
        self.restoreCursor()

    def focusOutEvent(self, ev):
        self.restoreCursor()

    def isCanvasVisible(self, shape):
        return self.visible.get(shape, True)

    def drawing(self):
        return self._mode == self.CREATE

    def editing(self):
        return self._mode == self.EDIT

    def setEditing(self, value=True):
        self._mode = self.EDIT if value else self.CREATE
        if not value:  # Create
            self.unHighlight()
            self.deSelectShape()

    def unHighlight(self):
        if self._hShape:
            self._hShape.highlightClear()
        self._hVertex = self._hShape = None

    def selectedVertex(self):
        return self._hVertex is not None

    def mouseMoveEvent(self, ev):
        """Update line with last point and current coordinates."""

        def drawPolygon(p):
            self.overrideCursor(CURSOR_DRAW)
            if not self._current:
                return

            if self.outOfPixmap(p):
                # Don't allow the user to draw outside the pixmap.
                # Project the point to the pixmap's edges.
                p = self.intersectionPoint(self._current[-1], p)
            elif len(self._current) > 1 and self.createMode in ['polygon', 'curve'] and \
                    self.closeEnough(p, self._current[0]):
                # Attract line to starting point and
                # colorise to alert the user.
                p = self._current[0]
                self.overrideCursor(CURSOR_POINT)
                self._current.highlightVertex(0, Shape.NEAR_VERTEX)
            self._current.update_cursor(p)
            self.repaint()
            self._current.highlightClear()

        def findHighlight(p):
            for shape in reversed([s for s in self.shapes if self.isCanvasVisible(s)]):
                # Look for a nearby vertex to highlight. If that fails,
                # check if we happen to be inside a shape.
                index = shape.nearestVertex(p, self._epsilon / self.scale)
                index_edge = shape.nearestEdge(p, self._epsilon / self.scale)
                if index is not None:
                    if self.selectedVertex():
                        self._hShape.highlightClear()
                    self._hVertex = index
                    self._hShape = shape
                    self._hEdge = index_edge
                    shape.highlightVertex(index, shape.MOVE_VERTEX)
                    self.overrideCursor(CURSOR_POINT)
                    self.setToolTip("Click & drag to move point")
                    self.setStatusTip(self.toolTip())
                    self.update()
                    break
                elif shape.containsPoint(p):
                    if self.selectedVertex():
                        self._hShape.highlightClear()
                    self._hVertex = None
                    self._hShape = shape
                    self._hEdge = index_edge
                    self.setToolTip(
                        "Click & drag to move shape '%s'" % shape.label)
                    self.setStatusTip(self.toolTip())
                    self.overrideCursor(CURSOR_GRAB)
                    self.update()
                    break
            else:  # Nothing found, clear highlights, reset state.
                if self._hShape:
                    self._hShape.highlightClear()
                    self.update()
                self._hVertex, self._hShape, self._hEdge = None, None, None
            self.edgeSelected.emit(self._hEdge is not None)

        try:
            if QT5:
                pos = self.transformPos(ev.localPos())
            else:
                pos = self.transformPos(ev.posF())
        except AttributeError:
            return

        # prevMovePoint should be an absolute position
        prevPos = self._prevMovePoint
        self._prevMovePoint = pos
        # Transform pointer location by image offset
        relPos = pos - self._imagePos
        self.restoreCursor()
        # Polygon drawing.
        if self.drawing():
            drawPolygon(relPos)
        # Polygon copy moving.
        elif QtCore.Qt.RightButton & ev.buttons():
            if self.selectedShapesCopy and self._prevPoint:
                self.overrideCursor(CURSOR_MOVE)
                self.boundedMoveShapes(self.selectedShapesCopy, relPos)
                self.repaint()
            elif self.selectedShapes:
                self.selectedShapesCopy = \
                    [s.copy() for s in self.selectedShapes]
                self.repaint()
        # Polygon/Vertex moving.
        elif QtCore.Qt.LeftButton & ev.buttons():
            if self.selectedVertex():
                self.boundedMoveVertex(relPos)
                self._movingShape = True
            elif self.selectedShapes and self._prevPoint:
                self.overrideCursor(CURSOR_MOVE)
                self.boundedMoveShapes(self.selectedShapes, relPos)
                self._movingShape = True
            # Nothing selected, drag canvas
            else:
                self.overrideCursor(CURSOR_MOVE)
                self._imagePos += pos - prevPos
            # Must repaint
            self.repaint()
        # Just hovering over the canvas, 2 posibilities:
        # - Highlight shapes
        # - Highlight vertex
        # Update shape/vertex fill and tooltip value accordingly.
        else:
            self.setToolTip("Image")
            self._movingShape = False
            findHighlight(relPos)

    def addPointToEdge(self):
        if (self._hShape is None and
                self._hEdge is None and
                self._prevMovePoint is None):
            return
        shape = self._hShape
        index = self._hEdge
        point = self._prevMovePoint
        shape.insertPoint(index, point)
        shape.highlightVertex(index, shape.MOVE_VERTEX)
        self._hShape = shape
        self._hVertex = index
        self._hEdge = None

    @property
    def join_shapes_dialog(self):
        """Construct a dialog on request to reflect the latest status."""
        join_dialog = JoinShapesDialog(parent_canvas=self)
        join_dialog.shapes_joined.connect(self.join_shapes)
        return join_dialog

    def join_shapes(self, shapes):
        self.shapes = list(set(self.shapes) - set(shapes))
        new_shape = MultiShape(shapes)
        self.shapes.append(new_shape)
        self.mergeShape.emit(shapes, new_shape)
        self.storeShapes()
        self.repaint()

    def mousePressEvent(self, ev):
        if QT5:
            pos = self.transformPos(ev.localPos())
        else:
            pos = self.transformPos(ev.posF())
        # Transform pointer location by image offset
        pos -= self._imagePos
        if ev.button() == QtCore.Qt.LeftButton:
            if self.drawing():
                if self._current:
                    # Add cursor position to existing shape.
                    ctrl_pressed = int(ev.modifiers()) == QtCore.Qt.ControlModifier
                    self._current.commit_point(cursor_point=pos)
                    if self._current.closed:
                        self.finalise()
                    elif self._current.complete and ctrl_pressed:
                        self.finalise()
                else:
                    # Out of bound, nothing to be done
                    if self.outOfPixmap(pos):
                        return
                    # Create new shape.
                    self._current = Shape(shape_type=self.createMode, init_point=pos)
                    if self.createMode == 'point':
                        self.finalise()
                    else:
                        self.setHiding()
                        self.drawingPolygon.emit(True)
                self.update()
            else:
                group_mode = (int(ev.modifiers()) == QtCore.Qt.ControlModifier)
                self.selectShapePoint(pos, multiple_selection_mode=group_mode)
                self._prevPoint = pos
                self.repaint()
        elif ev.button() == QtCore.Qt.RightButton and self.editing():
            group_mode = (int(ev.modifiers()) == QtCore.Qt.ControlModifier)
            self.selectShapePoint(pos, multiple_selection_mode=group_mode)
            self._prevPoint = pos
            self.repaint()

    def mouseReleaseEvent(self, ev):
        if ev.button() == QtCore.Qt.RightButton:
            menu = self.menus[len(self.selectedShapesCopy) > 0]
            self.restoreCursor()
            if not menu.exec_(self.mapToGlobal(ev.pos())) \
                    and self.selectedShapesCopy:
                # Cancel the move by deleting the shadow copy.
                self.selectedShapesCopy = []
                self.repaint()
        elif ev.button() == QtCore.Qt.LeftButton and self.selectedShapes:
            self.overrideCursor(CURSOR_GRAB)
        if self._movingShape:
            self.storeShapes()
            self.shapeMoved.emit()

    def endMove(self, copy):
        assert self.selectedShapes and self.selectedShapesCopy
        assert len(self.selectedShapesCopy) == len(self.selectedShapes)
        # del shape.fill_color
        # del shape.line_color
        if copy:
            for i, shape in enumerate(self.selectedShapesCopy):
                self.shapes.append(shape)
                self.selectedShapes[i] = shape
        else:
            for i, shape in enumerate(self.selectedShapesCopy):
                self.selectedShapes[i].points = shape.points
        self.selectedShapesCopy = []
        self.repaint()
        self.storeShapes()
        return True

    def setHiding(self, enable=True):
        self._hideBackground = self.hideBackground if enable else False

    def canCloseShape(self):
        return self.drawing() and self._current and len(self._current) > 2

    def mouseDoubleClickEvent(self, ev):
        # We need at least 4 points here, since the mousePress handler
        # adds an extra one before this handler is called.
        if self.canCloseShape() and len(self._current) > 3:
            self._current.undo_point()
            self.finalise()

    def selectShapes(self, shapes):
        self.setHiding()
        self.selectedShapes = shapes
        self.selectionChanged.emit()
        self.update()

    def selectShapePoint(self, point, multiple_selection_mode):
        """Select the first shape created which contains this point."""
        if self.selectedVertex():  # A vertex is marked for selection.
            index, shape = self._hVertex, self._hShape
            shape.highlightVertex(index, shape.MOVE_VERTEX)
        else:
            for shape in reversed(self.shapes):
                if self.isCanvasVisible(shape) and shape.containsPoint(point):
                    self.calculateOffsets(shape, point, margin=0.1)
                    self.setHiding()
                    if multiple_selection_mode:
                        if shape not in self.selectedShapes:
                            self.selectedShapes.append(shape)
                            self.selectionChanged.emit()
                    else:
                        self.selectedShapes = [shape]
                        self.selectionChanged.emit()
                    return
        self.deSelectShape()

    def calculateOffsets(self, shape, point, margin=0.0):
        rect = shape.boundingRect()
        x1 = rect.x() - point.x() - margin
        y1 = rect.y() - point.y() - margin
        x2 = rect.x() + rect.width() + margin - point.x()
        y2 = rect.y() + rect.height() + margin - point.y()
        self._offsets = QtCore.QPointF(x1, y1), QtCore.QPointF(x2, y2)

    def boundedMoveVertex(self, pos):
        index, shape = self._hVertex, self._hShape
        point = shape[index]
        if self.outOfPixmap(pos):
            assert not self.outOfPixmap(point)
            pos = self.intersectionPoint(point, pos)
        assert not self.outOfPixmap(pos)
        shape.moveVertexBy(index, pos - point)

    def boundedMoveShapes(self, shapes, pos):
        if self.outOfPixmap(pos):
            return False  # No need to move
        o1 = pos + self._offsets[0]
        if self.outOfPixmap(o1):
            pos -= QtCore.QPointF(min(0.0, o1.x()), min(0.0, o1.y()))
        o2 = pos + self._offsets[1]
        if self.outOfPixmap(o2):
            pos += QtCore.QPointF(min(0.0, self.pixmap.width() - o2.x()),
                                  min(0.0, self.pixmap.height() - o2.y()))
        # XXX: The next line tracks the new position of the cursor
        # relative to the shape, but also results in making it
        # a bit "shaky" when nearing the border and allows it to
        # go outside of the shape's area for some reason.
        # self.calculateOffsets(self.selectedShapes, pos)
        dp = pos - self._prevPoint
        if dp:
            for shape in shapes:
                shape.moveBy(dp)
            self._prevPoint = pos
            return True
        return False

    def deSelectShape(self):
        if self.selectedShapes:
            self.setHiding(False)
            self.selectedShapes = []
            self.selectionChanged.emit()
            self.update()

    def deleteSelected(self):
        deleted_shapes = []
        if self.selectedShapes:
            for shape in self.selectedShapes:
                self.shapes.remove(shape)
                deleted_shapes.append(shape)
            self.storeShapes()
            self.selectedShapes = []
            self.update()
        return deleted_shapes

    def copySelectedShapes(self):
        if self.selectedShapes:
            self.selectedShapesCopy = [s.copy() for s in self.selectedShapes]
            self.boundedShiftShapes(self.selectedShapesCopy)
            self.endMove(copy=True)
        return self.selectedShapes

    def boundedShiftShapes(self, shapes):
        # Try to move in one direction, and if it fails in another.
        # Give up if both fail.
        point = shapes[0][0]
        offset = QtCore.QPointF(2.0, 2.0)
        self._offsets = QtCore.QPointF(), QtCore.QPointF()
        self._prevPoint = point
        if not self.boundedMoveShapes(shapes, point - offset):
            self.boundedMoveShapes(shapes, point + offset)

    def paintEvent(self, event):
        if not self.pixmap:
            return super(Canvas, self).paintEvent(event)

        p = self._painter
        p.begin(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.setRenderHint(QtGui.QPainter.HighQualityAntialiasing)
        p.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)

        p.scale(self.scale, self.scale)
        p.translate(self.offsetToCenter())
        p.drawPixmap(self._imagePos, self.pixmap)
        # Translate by image position for all shapes
        # so that they move along
        p.translate(self._imagePos)
        Shape.scale = self.scale
        selected_shapes_set = set(self.selectedShapes)
        for shape in self.shapes:
            selected = shape in selected_shapes_set
            if (selected or not self._hideBackground) and self.isCanvasVisible(shape):
                shape.paint(p, fill=(selected or shape == self._hShape))
        if self._current:
            self._current.paint(p)
        if self.selectedShapesCopy:
            for s in self.selectedShapesCopy:
                s.paint(p)

        if (self.fillDrawing() and self.createMode == 'polygon' and
                self._current is not None and len(self._current.points) >= 2):
            drawing_shape = self._current.copy()
            drawing_shape.fill = True
            drawing_shape.fill_color.setAlpha(64)
            drawing_shape.paint(p)

        p.end()

    def transformPos(self, point):
        """Convert from widget-logical coordinates to painter-logical ones."""
        return point / self.scale - self.offsetToCenter()

    def offsetToCenter(self):
        s = self.scale
        area = super(Canvas, self).size()
        w, h = self.pixmap.width() * s, self.pixmap.height() * s
        aw, ah = area.width(), area.height()
        x = (aw - w) / (2 * s) if aw > w else 0
        y = (ah - h) / (2 * s) if ah > h else 0
        return QtCore.QPointF(x, y)

    def outOfPixmap(self, p):
        w, h = self.pixmap.width(), self.pixmap.height()
        return not (0 <= p.x() <= w and 0 <= p.y() <= h)

    def finalise(self):
        assert self._current
        self._current.finalize()
        self.shapes.append(self._current)
        self.storeShapes()
        self._current = None
        self.setHiding(False)
        self.newShape.emit()
        self.update()

    def closeEnough(self, p1, p2):
        # d = distance(p1 - p2)
        # m = (p1-p2).manhattanLength()
        # print "d %.2f, m %d, %.2f" % (d, m, d - m)
        # divide by scale to allow more precision when zoomed in
        return labelme.utils.distance(p1 - p2) < (self._epsilon / self.scale)

    def intersectionPoint(self, p1, p2):
        # Cycle through each image edge in clockwise fashion,
        # and find the one intersecting the current line segment.
        # http://paulbourke.net/geometry/lineline2d/
        size = self.pixmap.size()
        points = [(0, 0),
                  (size.width(), 0),
                  (size.width(), size.height()),
                  (0, size.height())]
        x1, y1 = p1.x(), p1.y()
        x2, y2 = p2.x(), p2.y()
        d, i, (x, y) = min(self.intersectingEdges((x1, y1), (x2, y2), points))
        x3, y3 = points[i]
        x4, y4 = points[(i + 1) % 4]
        if (x, y) == (x1, y1):
            # Handle cases where previous point is on one of the edges.
            if x3 == x4:
                return QtCore.QPointF(x3, min(max(0, y2), max(y3, y4)))
            else:  # y3 == y4
                return QtCore.QPointF(min(max(0, x2), max(x3, x4)), y3)
        return QtCore.QPointF(x, y)

    @staticmethod
    def intersectingEdges(point1, point2, points):
        """Find intersecting edges.

        For each edge formed by `points', yield the intersection
        with the line segment `(x1,y1) - (x2,y2)`, if it exists.
        Also return the distance of `(x2,y2)' to the middle of the
        edge along with its index, so that the one closest can be chosen.
        """
        (x1, y1) = point1
        (x2, y2) = point2
        for i in range(4):
            x3, y3 = points[i]
            x4, y4 = points[(i + 1) % 4]
            denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
            nua = (x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)
            nub = (x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)
            if denom == 0:
                # This covers two cases:
                #   nua == nub == 0: Coincident
                #   otherwise: Parallel
                continue
            ua, ub = nua / denom, nub / denom
            if 0 <= ua <= 1 and 0 <= ub <= 1:
                x = x1 + ua * (x2 - x1)
                y = y1 + ua * (y2 - y1)
                m = QtCore.QPointF((x3 + x4) / 2, (y3 + y4) / 2)
                d = labelme.utils.distance(m - QtCore.QPointF(x2, y2))
                yield d, i, (x, y)

    # These two, along with a call to adjustSize are required for the
    # scroll area.
    def sizeHint(self):
        return self.minimumSizeHint()

    def minimumSizeHint(self):
        if self.pixmap:
            return self.scale * self.pixmap.size()
        return super(Canvas, self).minimumSizeHint()

    def wheelEvent(self, ev):
        if QT5:
            mods = ev.modifiers()
            delta = ev.angleDelta()
            if QtCore.Qt.ControlModifier == int(mods):
                # with Ctrl/Command key
                # zoom
                self.zoomRequest.emit(delta.y(), ev.pos())
            else:
                # scroll
                self.scrollRequest.emit(delta.x(), QtCore.Qt.Horizontal)
                self.scrollRequest.emit(delta.y(), QtCore.Qt.Vertical)
        else:
            if ev.orientation() == QtCore.Qt.Vertical:
                mods = ev.modifiers()
                if QtCore.Qt.ControlModifier == int(mods):
                    # with Ctrl/Command key
                    self.zoomRequest.emit(ev.delta(), ev.pos())
                else:
                    self.scrollRequest.emit(
                        ev.delta(),
                        QtCore.Qt.Horizontal
                        if (QtCore.Qt.ShiftModifier == int(mods))
                        else QtCore.Qt.Vertical)
            else:
                self.scrollRequest.emit(ev.delta(), QtCore.Qt.Horizontal)
        ev.accept()

    def keyPressEvent(self, ev):
        key = ev.key()
        if key == QtCore.Qt.Key_Escape and self._current:
            self._current = None
            self.drawingPolygon.emit(False)
            self.update()
        elif key == QtCore.Qt.Key_Return and self.canCloseShape():
            self.finalise()

    def setLabelFor(self, shape, text, form):
        def argbToQColor(val):
            hexStr = hex(val)[2:]
            a, r, g, b = tuple(int(hexStr[i:i + 2], 16) for i in (0, 2, 4, 6))
            return QtGui.QColor(r, g, b, a)

        shape.label = text
        shape.form = form
        shape.line_color = shape.fill_color = argbToQColor(self._label_color[text])

    def setLastLabel(self, text, form):
        self.setLabelFor(self.shapes[-1], text, form)
        self.shapesBackups.pop()
        self.storeShapes()
        return self.shapes[-1]

    def undoLastLine(self):
        assert self.shapes
        self._current = self.shapes.pop()
        self._current.undo_point(forced=True)
        if self._current.is_empty():
            self._current = None
        else:
            self.drawingPolygon.emit(True)

    def undoLastPoint(self):
        if not self._current:
            return
        if not self._current.undo_point():
            return
        if self._current.is_empty():
            self._current = None
            self.drawingPolygon.emit(False)
        self.repaint()

    def load_image_file(self, image_file):
        self._image_file = image_file
        self.pixmap = image_file.pixmap
        if not Config.get('keep_prev'):
            self.shapes = []
        self.repaint()

    def loadShapes(self, shapes, replace=True):
        if replace:
            self.shapes = list(shapes)
        else:
            self.shapes.extend(shapes)
        self.storeShapes()
        self._current = None
        self.repaint()

    def setShapeVisible(self, shape, value):
        self.visible[shape] = value
        self.repaint()

    def overrideCursor(self, cursor):
        self.restoreCursor()
        self._cursor = cursor
        QtWidgets.QApplication.setOverrideCursor(cursor)

    @staticmethod
    def restoreCursor():
        QtWidgets.QApplication.restoreOverrideCursor()

    def resetState(self):
        self.restoreCursor()
        self.pixmap = None
        self.shapesBackups = []
        self.update()
