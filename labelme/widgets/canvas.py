from math import sqrt

from qtpy import QtWidgets
from qtpy.QtCore import Qt, Signal, QPointF, QRectF
from qtpy.QtGui import QPainter

import labelme.utils
from labelme.backend import ModelLoader
from labelme.custom_widgets import PreviewCanvas, LoadingDialog
from labelme.shape import Shape, PoseShape, EditingShape


CURSOR_DEFAULT = Qt.ArrowCursor
CURSOR_POINT = Qt.PointingHandCursor
CURSOR_DRAW = Qt.CrossCursor
CURSOR_MOVE = Qt.ClosedHandCursor
CURSOR_GRAB = Qt.OpenHandCursor


class Canvas(PreviewCanvas):
    newShape = Signal()
    mergeShape = Signal(list, PoseShape)
    shapeMoved = Signal()
    drawingPolygon = Signal(bool)

    CREATE, EDIT = 0, 1

    # polygon, rectangle, line, or point
    _createMode = 'polygon'
    _fill_drawing = False
    _freeform_dist = 0.5

    def __init__(self, *args, **kwargs):
        super(Canvas, self).__init__(*args, **kwargs)
        # Initialise local state.
        self._mode = self.EDIT
        self._current = None
        self._prevPoint, self._prevMovePoint = QPointF(), QPointF()
        self._movingShape = False
        # Loading dialog
        self.loading_dialog = LoadingDialog(self)
        # DL models
        self.model_loader = None
        # Public state
        self.shapesBackups = []
        # Menus:
        # 0: right-click without selection and dragging of shapes
        # 1: right-click with selection and dragging of shapes
        self.menus = (QtWidgets.QMenu(), QtWidgets.QMenu())

    def to_preview_canvas(self):
        preview = PreviewCanvas()
        preview.load_image_file(self._image_file)
        preview.loadShapes(self.shapes.copy())
        return preview

    def fillDrawing(self):
        return self._fill_drawing

    def load_models(self):
        def on_model_inited(succeeded, model):
            self.model_loader = model
            self.loading_dialog.loaded.emit(succeeded)

        ModelLoader.main_thread_ctor(on_model_inited)
        self.loading_dialog.exec_()

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
        self.load_image_file(value['image_file'], load_model_hint=False)
        self.loadShapes(value['shapes'])

    def storeShapes(self):
        from copy import deepcopy
        if len(self.shapesBackups) >= 10:
            self.shapesBackups = self.shapesBackups[-9:]
        self.shapesBackups.append(deepcopy(self.shapes))

    @property
    def isShapeRestorable(self):
        if len(self.shapesBackups) < 2:
            return False
        return True

    def restoreShape(self):
        from copy import deepcopy
        if not self.isShapeRestorable:
            return False
        self.shapesBackups.pop()  # latest
        self.shapes = deepcopy(self.shapesBackups[-1])
        self.selectedShapes = []
        self.repaint()
        return True

    def drawing(self):
        return self._mode == self.CREATE

    def editing(self):
        return self._mode == self.EDIT

    def setEditing(self, value=True):
        self._mode = self.EDIT if value else self.CREATE
        if value:
            self._current = None
            self.drawingPolygon.emit(False)
            self.update()
        else:  # Create
            self.unHighlight()
            self.deSelectShape()

    def mouseMoveEvent(self, ev):
        """Update line with last point and current coordinates."""

        def drawPolygon(p, last_p):
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
            if self.createMode == 'freeform' and Qt.LeftButton & ev.buttons():
                diff = p - last_p
                dist = sqrt(diff.x() ** 2 + diff.y() ** 2)
                if dist > Canvas._freeform_dist:
                    self._current.add_point()
            self._current.highlightClear()
            self.repaint()
            self._current.highlightClear()

        try:
            pos = self.transformPos(ev.localPos())
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
            drawPolygon(relPos, prevPos - self._imagePos)
        # Polygon copy moving.
        elif Qt.RightButton & ev.buttons():
            if self.selectedShapesCopy and self._prevPoint:
                self.overrideCursor(CURSOR_MOVE)
                self.boundedMoveShapes(self.selectedShapesCopy, relPos)
                self.repaint()
            elif self.selectedShapes:
                self.selectedShapesCopy = \
                    [s.copy() for s in self.selectedShapes]
                self.repaint()
        # Polygon/Vertex moving.
        elif Qt.LeftButton & ev.buttons():
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
            self.findHighlight(relPos)

    def addPointToEdge(self):
        if (self._hShape is None and
                self._hEdge is None and
                self._prevMovePoint is None):
            return
        shape = self._hShape
        index = self._hEdge
        point = self._prevMovePoint
        shape.insert_point(index, point)
        shape.highlightVertex(index, shape.MOVE_VERTEX)
        self._hShape = shape
        self._hVertex = index
        self._hEdge = None

    def mousePressEvent(self, ev):
        try:
            pos = self.transformPos(ev.localPos())
        except AttributeError:
            return

        # Transform pointer location by image offset
        pos -= self._imagePos
        if ev.button() == Qt.LeftButton:
            if self.drawing():
                if self._current:
                    # Add cursor position to existing shape.
                    ctrl_pressed = int(ev.modifiers()) == Qt.ControlModifier
                    if self._current.closed:
                        self.finalise()
                    elif self._current.complete and ctrl_pressed:
                        self.finalise()
                    else:
                        self._current.add_point()
                else:
                    # Out of bound, nothing to be done
                    if self.outOfPixmap(pos):
                        return
                    # Create new shape.
                    self._current = EditingShape(self.createMode, pos)
                    if self.createMode == 'point':
                        self.finalise()
                    else:
                        self._current.add_point()
                        self.drawingPolygon.emit(True)
                self.update()
            else:
                group_mode = (int(ev.modifiers()) == Qt.ControlModifier)
                self.selectShapePoint(pos, multiple_selection_mode=group_mode)
                self._prevPoint = pos
                self.repaint()
        elif ev.button() == Qt.RightButton and self.editing():
            group_mode = (int(ev.modifiers()) == Qt.ControlModifier)
            self.selectShapePoint(pos, multiple_selection_mode=group_mode)
            self._prevPoint = pos
            self.repaint()

    def mouseReleaseEvent(self, ev):
        if ev.button() == Qt.RightButton:
            menu = self.menus[len(self.selectedShapesCopy) > 0]
            self.restoreCursor()
            if not menu.exec_(self.mapToGlobal(ev.pos())) \
                    and self.selectedShapesCopy:
                # Cancel the move by deleting the shadow copy.
                self.selectedShapesCopy = []
                self.repaint()
        elif ev.button() == Qt.LeftButton:
            if self.selectedShapes:
                self.overrideCursor(CURSOR_GRAB)
            elif self.createMode == 'freeform':
                self.finalise()
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

    def canCloseShape(self):
        return self.drawing() and self._current and len(self._current) > 2

    def mouseDoubleClickEvent(self, ev):
        # We need at least 4 points here, since the mousePress handler
        # adds an extra one before this handler is called.
        if self.canCloseShape() and len(self._current) > 3:
            self._current.undo_point()
            self.finalise()

    def selectShapes(self, shapes):
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
                    if multiple_selection_mode:
                        if shape not in self.selectedShapes:
                            self.selectedShapes.append(shape)
                            self.selectionChanged.emit()
                    else:
                        self.selectedShapes = [shape]
                        self.selectionChanged.emit()
                    return
        self.deSelectShape()

    def boundedMoveVertex(self, pos):
        index, shape = self._hVertex, self._hShape
        point = shape[index]
        if self.outOfPixmap(pos):
            assert not self.outOfPixmap(point)
            pos = self.intersectionPoint(point, pos)
        assert not self.outOfPixmap(pos)
        shape.moveVertexBy(index, pos - point)

    def boundedMoveShapes(self, shapes, pos):
        def get_shapes_bounding_rect(shapes, margin=0.0):
            rects = [shape.boundingRect() for shape in shapes]
            reduced = QRectF()
            for rect in rects:
                reduced = reduced.united(rect)
            return reduced.adjusted(-margin, -margin, margin, margin)

        def get_shift_amount(left, right):
            if left > 0:
                return left
            elif right < 0:
                return right
            else:
                return 0

        if self.outOfPixmap(pos):
            return False  # No need to move
        rect = get_shapes_bounding_rect(shapes, margin=0.1)
        move_vector = pos - self._prevPoint
        if not move_vector:
            return False
        rect.moveCenter(rect.center() + move_vector)
        rect_xy = rect.getCoords()
        pixmap_xy = 0, 0, self.pixmap.width(), self.pixmap.height()
        dx1, dy1, dx2, dy2 = [pxy - rxy for pxy, rxy in zip(pixmap_xy, rect_xy)]
        move_vector += QPointF(
            get_shift_amount(dx1, dx2),
            get_shift_amount(dy1, dy2)
        )
        for shape in shapes:
            shape.moveBy(move_vector)
        self._prevPoint = pos
        return True

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
        import jsonpickle
        return jsonpickle.encode((
            self._image_file.filename,
            self.selectedShapes
        ))

    def pasteShapes(self, shapes_json):
        import jsonpickle
        source_filename, shapes = jsonpickle.decode(shapes_json)
        if source_filename == self._image_file.filename:
            self.boundedShiftShapes(shapes)
        self.loadShapes(shapes, replace=False)
        self.selectedShapes = shapes
        self.repaint()
        return shapes

    def boundedShiftShapes(self, shapes, margin=5.0):
        # Try to move in one direction, and if it fails in another.
        # Give up if both fail.
        point = shapes[0][0]
        offset = QPointF(margin, margin)
        self._prevPoint = point
        if not self.boundedMoveShapes(shapes, point - offset):
            self.boundedMoveShapes(shapes, point + offset)

    def paintEvent(self, event):
        if not self.pixmap:
            return super(Canvas, self).paintEvent(event)

        p = self._painter
        p.begin(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.HighQualityAntialiasing)
        p.setRenderHint(QPainter.SmoothPixmapTransform)

        p.scale(self.scale, self.scale)
        p.translate(self.offsetToCenter())
        p.drawPixmap(self._imagePos, self.pixmap)
        # Translate by image position for all shapes
        # so that they move along
        p.translate(self._imagePos)
        selected_shapes_set = set(self.selectedShapes)
        for shape in self.shapes:
            selected = shape in selected_shapes_set
            if self.isCanvasVisible(shape):
                shape.paint(
                    p, fill=(selected or shape == self._hShape), canvas=self
                )
        if self._current:
            self._current.paint(p, canvas=self)
        if self.selectedShapesCopy:
            for s in self.selectedShapesCopy:
                s.paint(p, canvas=self)

        if (self.fillDrawing() and self.createMode == 'polygon' and
                self._current is not None and len(self._current.points) >= 2):
            drawing_shape = self._current.copy()
            drawing_shape.fill = True
            drawing_shape.fill_color.setAlpha(64)
            drawing_shape.paint(p, canvas=self)

        p.end()

    def finalise(self):
        assert self._current
        result = self._current.to_immutable_point()
        if not result:
            return
        self.shapes.append(result)
        self.storeShapes()
        self._current = None
        self.repaint()
        self.newShape.emit()
        self.update()

    def closeEnough(self, p1, p2):
        # d = distance(p1 - p2)
        # m = (p1-p2).manhattanLength()
        # print "d %.2f, m %d, %.2f" % (d, m, d - m)
        # divide by scale to allow more precision when zoomed in
        return labelme.utils.distance(p1 - p2) < (self.epsilon / self.scale)

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
                return QPointF(x3, min(max(0, y2), max(y3, y4)))
            else:  # y3 == y4
                return QPointF(min(max(0, x2), max(x3, x4)), y3)
        return QPointF(x, y)

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
                m = QPointF((x3 + x4) / 2, (y3 + y4) / 2)
                d = labelme.utils.distance(m - QPointF(x2, y2))
                yield d, i, (x, y)

    def keyPressEvent(self, ev):
        key = ev.key()
        if key == Qt.Key_Escape and self._current:
            self._current = None
            self.drawingPolygon.emit(False)
            self.update()
        elif key == Qt.Key_Return and self.canCloseShape():
            self.finalise()

    def setLastLabel(self, form, shapes):
        if shapes:
            self.loadShapes(shapes)
        else:
            self.shapes[-1].set_label(form)
        self.shapesBackups.pop()
        self.storeShapes()
        return self.shapes[-1]

    def undoLastLine(self):
        assert self.shapes
        self._current = EditingShape.undo_into_editing_point(self.shapes.pop())
        if self._current:
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

    def load_image_file(self, image_file, load_model_hint=True):
        def on_model_result_received(succeeded, model_data):
            self.loading_dialog.loaded.emit(succeeded)
            if succeeded:
                self.shapes.extend(model_data)

        super().load_image_file(image_file)
        if load_model_hint:
            if not self.model_loader:
                self.load_models()
            self.model_loader.main_thread_infer(
                image_file.filename, on_completion=on_model_result_received
            )
            self.loading_dialog.exec_()
        self.repaint()

    def loadShapes(self, shapes, replace=True):
        super().loadShapes(shapes, replace)
        self.storeShapes()
        self._current = None
        self.repaint()

    def resetState(self):
        self.restoreCursor()
        self.shapesBackups = []
        self.selectedShapes = []
        self.update()
