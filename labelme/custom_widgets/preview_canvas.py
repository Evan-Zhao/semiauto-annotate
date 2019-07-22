from qtpy.QtCore import Qt, Signal, QPointF
from qtpy.QtGui import QPainter
from qtpy.QtWidgets import QWidget, QApplication

from labelme.utils import Config

CURSOR_DEFAULT = Qt.ArrowCursor
CURSOR_POINT = Qt.PointingHandCursor
CURSOR_GRAB = Qt.OpenHandCursor


# TODO(unknown):
# - [maybe] Find optimal epsilon value.

class PreviewCanvas(QWidget):
    zoomRequest = Signal(int, QPointF)
    scrollRequest = Signal(int, int)
    selectionChanged = Signal()
    vertexSelected = Signal()
    edgeSelected = Signal(bool)
    min_size, max_size = 400, 1000

    def __init__(self, no_highlight=False, *args, **kwargs):
        super(PreviewCanvas, self).__init__(*args, **kwargs)
        # Initialise local state.
        self._no_highlight = no_highlight
        self._image_file = None
        self._hShape, self._hVertex, self._hEdge = None, None, None
        self._painter = QPainter()
        self._cursor = CURSOR_DEFAULT
        self._imagePos, self.scale = QPointF(), 1.0
        self._painter = QPainter()
        # Public state
        self.shapes = []
        self.selectedShapes = []
        self.selectedShapesCopy = []
        self.visible = {}
        # Set widget options.
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.WheelFocus)

    @property
    def epsilon(self):
        return Config.get('epsilon', default=10.0)

    @property
    def pixmap(self):
        return self._image_file.pixmap if self._image_file else None

    def is_empty(self):
        return self.pixmap is None

    def has_shapes(self):
        return bool(self.shapes)

    def enterEvent(self, ev):
        self.overrideCursor(self._cursor)

    def leaveEvent(self, ev):
        self.restoreCursor()

    def focusOutEvent(self, ev):
        self.restoreCursor()

    def unHighlight(self):
        if self._hShape:
            self._hShape.highlightClear()
        self._hVertex = self._hShape = None

    def selectedVertex(self):
        return self._hVertex is not None

    @staticmethod
    def compute_scale(current_scale, w, h):
        min_wh, max_wh = min(w, h), max(w, h)
        min_scale = PreviewCanvas.min_size / min_wh
        max_scale = PreviewCanvas.max_size / max_wh
        if min_scale > current_scale:
            return min_scale
        elif max_scale < current_scale:
            return max_scale
        return current_scale

    # These two, along with a call to adjustSize are required for the
    # scroll area.
    def sizeHint(self):
        return self.minimumSizeHint()

    def minimumSizeHint(self):
        if self.pixmap:
            return self.scale * self.pixmap.size()
        return super().minimumSizeHint()

    def isCanvasVisible(self, shape):
        return self.visible.get(shape, True)

    def setShapeVisible(self, shape, value):
        self.visible[shape] = value
        self.repaint()

    def load_image_file(self, image_file, **kwargs):
        self._image_file = image_file
        if not Config.get('keep_prev'):
            self.shapes = []
        pixmap = image_file.pixmap
        self.scale = self.compute_scale(self.scale, pixmap.width(), pixmap.height())
        self.repaint()

    def overrideCursor(self, cursor):
        self.restoreCursor()
        self._cursor = cursor
        QApplication.setOverrideCursor(cursor)

    @staticmethod
    def restoreCursor():
        QApplication.restoreOverrideCursor()

    def findHighlight(self, p):
        for shape in reversed([s for s in self.shapes if self.isCanvasVisible(s)]):
            # Look for a nearby vertex to highlight. If that fails,
            # check if we happen to be inside a shape.
            index = shape.nearestVertex(p, self.epsilon / self.scale)
            index_edge = shape.nearestEdge(p, self.epsilon / self.scale)
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

    def mouseMoveEvent(self, ev):
        try:
            pos = self.transformPos(ev.localPos())
        except AttributeError:
            return

        # Transform pointer location by image offset
        rel_pos = pos - self._imagePos
        # Just hovering over the canvas, 2 posibilities:
        # - Highlight shapes
        # - Highlight vertex
        # Update shape/vertex fill and tooltip value accordingly.
        self.setToolTip("Image")
        self.findHighlight(rel_pos)

    def mousePressEvent(self, ev):
        try:
            pos = self.transformPos(ev.localPos())
        except AttributeError:
            return
        # Transform pointer location by image offset
        pos -= self._imagePos
        if ev.button() == Qt.LeftButton:
            group_mode = (int(ev.modifiers()) == Qt.ControlModifier)
            if self.selectedVertex():  # A vertex is marked for selection.
                index, shape = self._hVertex, self._hShape
                shape.highlightVertex(index, shape.MOVE_VERTEX)
                self.vertexSelected.emit()
            self.select_shape(pos, multiple_selection_mode=group_mode)
            self.repaint()

    def select_shape(self, point, multiple_selection_mode):
        """Select the first shape created which contains this point."""
        for shape in reversed(self.shapes):
            if not shape.containsPoint(point):
                continue
            if multiple_selection_mode:
                if shape not in self.selectedShapes:
                    self.selectedShapes.append(shape)
                    self.selectionChanged.emit()
            else:
                self.selectedShapes = [shape]
                self.selectionChanged.emit()
            return
        # If clicked on nothing, deselect all.
        self.deSelectShape()

    def deSelectShape(self):
        if self.selectedShapes:
            self.selectedShapes = []
            self.selectionChanged.emit()
            self.update()

    def paintEvent(self, event):
        if not self.pixmap:
            return super(PreviewCanvas, self).paintEvent(event)

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
        p.end()

    def transformPos(self, point):
        """Convert from widget-logical coordinates to painter-logical ones."""
        return point / self.scale - self.offsetToCenter()

    def offsetToCenter(self):
        s = self.scale
        area = super(PreviewCanvas, self).size()
        w, h = self.pixmap.width() * s, self.pixmap.height() * s
        aw, ah = area.width(), area.height()
        x = (aw - w) / (2 * s) if aw > w else 0
        y = (ah - h) / (2 * s) if ah > h else 0
        return QPointF(x, y)

    def outOfPixmap(self, p):
        w, h = self.pixmap.width(), self.pixmap.height()
        return not (0 <= p.x() <= w and 0 <= p.y() <= h)

    def wheelEvent(self, ev):
        mods = ev.modifiers()
        delta = ev.angleDelta()
        if Qt.ControlModifier == int(mods):
            # with Ctrl/Command key
            # zoom
            self.zoomRequest.emit(delta.y(), ev.pos())
        else:
            # scroll
            self.scrollRequest.emit(delta.x(), Qt.Horizontal)
            self.scrollRequest.emit(delta.y(), Qt.Vertical)
        ev.accept()

    def loadShapes(self, shapes, replace=True):
        if replace:
            self.shapes = list(shapes)
        else:
            self.shapes.extend(shapes)
