from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets

from labelme import QT5
from labelme.shape import Shape
from labelme.utils import Config


class PreviewCanvas(QtWidgets.QWidget):
    zoomRequest = QtCore.Signal(int, QtCore.QPointF)
    scrollRequest = QtCore.Signal(int, int)
    selectionChanged = QtCore.Signal()
    vertexSelected = QtCore.Signal()
    min_size, max_size = 400, 1000

    def __init__(self, no_highlight=False, *args, **kwargs):
        self._epsilon = Config.get('epsilon', default=10.0)
        self._label_color = Config.get('label_color')
        super(PreviewCanvas, self).__init__(*args, **kwargs)
        # Initialise local state.
        self._no_highlight = no_highlight
        self.shapes = []
        self.selected_shapes = set()  # save the selected shapes here
        self.hShape, self.hVertex = None, None
        self._imagePos, self._scale = QtCore.QPointF(), 1.0
        self._pixmap = QtGui.QPixmap()
        self._painter = QtGui.QPainter()
        # Set widget options.
        self.setMouseTracking(True)
        self.setFocusPolicy(QtCore.Qt.WheelFocus)

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

    def set_pixmap(self, pixmap):
        self._pixmap = pixmap
        self._scale = self.compute_scale(self._scale, pixmap.width(), pixmap.height())

    @classmethod
    def from_canvas(cls, canvas, *args, **kwargs):
        preview_canvas = cls(*args, **kwargs)
        preview_canvas.shapes = canvas.shapes
        preview_canvas.set_pixmap(canvas.pixmap)
        return preview_canvas

    def set_selection_signal(self, selection_changed):
        """
        Let canvas respond to selection change requests
        (so as to bind to label lists, etc.)
        """
        selection_changed.connect(self.on_selection_requested)

    def on_selection_requested(self, selected_items):
        """Handles external change of selection."""
        self.selected_shapes = selected_items
        self.repaint()

    def selectedVertex(self):
        return self.hVertex is not None

    def mouseMoveEvent(self, ev):
        try:
            if QT5:
                pos = self.transformPos(ev.localPos())
            else:
                pos = self.transformPos(ev.posF())
        except AttributeError:
            return

        # Transform pointer location by image offset
        rel_pos = pos - self._imagePos
        # Just hovering over the canvas, 2 posibilities:
        # - Highlight shapes
        # - Highlight vertex
        # Update shape/vertex fill and tooltip value accordingly.
        for shape in reversed([s for s in self.shapes]):
            # Look for a nearby vertex to highlight. If that fails,
            # check if we happen to be inside a shape.
            index = shape.nearestVertex(rel_pos, self._epsilon / self._scale)
            if index is not None:
                if self.selectedVertex():
                    self.hShape.highlightClear()
                self.hVertex = index
                self.hShape = shape
                shape.highlightVertex(index, shape.MOVE_VERTEX)
                self.update()
                break
            elif shape.containsPoint(rel_pos):
                if self.selectedVertex():
                    self.hShape.highlightClear()
                self.hVertex = None
                self.hShape = shape
                self.setToolTip("Click to select shape '%s'" % shape.label)
                self.setStatusTip(self.toolTip())
                self.update()
                break
        else:  # Nothing found, clear highlights, reset state.
            if self.hShape:
                self.hShape.highlightClear()
                self.update()
            self.hVertex, self.hShape = None, None

    def mousePressEvent(self, ev):
        if QT5:
            pos = self.transformPos(ev.localPos())
        else:
            pos = self.transformPos(ev.posF())
        # Transform pointer location by image offset
        pos -= self._imagePos
        if ev.button() == QtCore.Qt.LeftButton:
            group_mode = (int(ev.modifiers()) == QtCore.Qt.ControlModifier)
            if self.selectedVertex():  # A vertex is marked for selection.
                index, shape = self.hVertex, self.hShape
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
                if shape not in self.selected_shapes:
                    self.selected_shapes.add(shape)
                    self.selectionChanged.emit()
            else:
                self.selected_shapes = {shape}
                self.selectionChanged.emit()
            return
        # If clicked on nothing, deselect all.
        self.deSelectShape()

    def deSelectShape(self):
        if self.selected_shapes:
            self.selected_shapes = set()
            self.selectionChanged.emit()
            self.update()

    def paintEvent(self, event):
        if not self._pixmap:
            return super(PreviewCanvas, self).paintEvent(event)

        p = self._painter
        p.begin(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.setRenderHint(QtGui.QPainter.HighQualityAntialiasing)
        p.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)

        p.scale(self._scale, self._scale)
        p.translate(self.offsetToCenter())
        p.drawPixmap(self._imagePos, self._pixmap)
        # Translate by image position for all shapes
        # so that they move along
        p.translate(self._imagePos)
        Shape.scale = self._scale
        for shape in self.shapes:
            fill = (not self._no_highlight and (
                    shape in self.selected_shapes or shape == self.hShape
            ))
            shape.paint(p, fill=fill)
        p.end()

    def transformPos(self, point):
        """Convert from widget-logical coordinates to painter-logical ones."""
        return point / self._scale - self.offsetToCenter()

    def offsetToCenter(self):
        s = self._scale
        area = super(PreviewCanvas, self).size()
        w, h = self._pixmap.width() * s, self._pixmap.height() * s
        aw, ah = area.width(), area.height()
        x = (aw - w) / (2 * s) if aw > w else 0
        y = (ah - h) / (2 * s) if ah > h else 0
        return QtCore.QPointF(x, y)

    # These two, along with a call to adjustSize are required for the
    # scroll area.
    def sizeHint(self):
        return self.minimumSizeHint()

    def minimumSizeHint(self):
        if self._pixmap:
            return self._scale * self._pixmap.size()
        return super(PreviewCanvas, self).minimumSizeHint()

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

    def replace_and_focus_shape(self, shape: Shape, padding: float = 40.0):
        shape_rect = shape.boundingRect()
        scale = self.compute_scale(self._scale, shape_rect.width(), shape_rect.height())
        pd = padding / scale
        shape_rect = shape_rect.adjusted(-pd, -pd, pd, pd)
        rect_int = QtCore.QRect(*shape_rect.getRect())
        self.set_pixmap(self._pixmap.copy(rect_int))
        shape = shape.copy()
        shape.moveBy(-rect_int.topLeft())
        self.shapes = [shape]
        self.repaint()
        return shape
