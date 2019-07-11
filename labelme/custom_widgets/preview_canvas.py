from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets

from labelme import QT5
from labelme.shape import Shape


class PreviewCanvas(QtWidgets.QWidget):
    zoomRequest = QtCore.Signal(int, QtCore.QPointF)
    scrollRequest = QtCore.Signal(int, int)
    selectionChanged = QtCore.Signal()

    # polygon, rectangle, line, or point
    _createMode = 'polygon'
    _fill_drawing = False

    def __init__(self, *args, **kwargs):
        self.epsilon = kwargs.pop('epsilon', 10.0)
        self.label_color = kwargs.pop('label_color')
        super(PreviewCanvas, self).__init__(*args, **kwargs)
        # Initialise local state.
        self.shapes = []
        self.selected_shapes = set()  # save the selected shapes here
        self.imagePos = QtCore.QPointF()
        self.scale = 1.0
        self.pixmap = QtGui.QPixmap()
        self.hShape = None
        self.hVertex = None
        self.hEdge = None
        self._painter = QtGui.QPainter()
        # Set widget options.
        self.setMouseTracking(True)
        self.setFocusPolicy(QtCore.Qt.WheelFocus)

    @classmethod
    def from_canvas(cls, canvas):
        preview_canvas = cls(epsilon=canvas.epsilon, label_color=canvas.label_color)
        preview_canvas.shapes = canvas.shapes
        preview_canvas.pixmap = canvas.pixmap
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
        rel_pos = pos - self.imagePos
        # Just hovering over the canvas, 2 posibilities:
        # - Highlight shapes
        # - Highlight vertex
        # Update shape/vertex fill and tooltip value accordingly.
        for shape in reversed([s for s in self.shapes]):
            # Look for a nearby vertex to highlight. If that fails,
            # check if we happen to be inside a shape.
            index = shape.nearestVertex(rel_pos, self.epsilon / self.scale)
            index_edge = shape.nearestEdge(rel_pos, self.epsilon / self.scale)
            if index is not None:
                if self.selectedVertex():
                    self.hShape.highlightClear()
                self.hVertex = index
                self.hShape = shape
                self.hEdge = index_edge
                shape.highlightVertex(index, shape.MOVE_VERTEX)
                self.update()
                break
            elif shape.containsPoint(rel_pos):
                if self.selectedVertex():
                    self.hShape.highlightClear()
                self.hVertex = None
                self.hShape = shape
                self.hEdge = index_edge
                self.setToolTip("Click to select shape '%s'" % shape.label)
                self.setStatusTip(self.toolTip())
                self.update()
                break
        else:  # Nothing found, clear highlights, reset state.
            if self.hShape:
                self.hShape.highlightClear()
                self.update()
            self.hVertex, self.hShape, self.hEdge = None, None, None

    def mousePressEvent(self, ev):
        if QT5:
            pos = self.transformPos(ev.localPos())
        else:
            pos = self.transformPos(ev.posF())
        # Transform pointer location by image offset
        pos -= self.imagePos
        if ev.button() == QtCore.Qt.LeftButton:
            group_mode = (int(ev.modifiers()) == QtCore.Qt.ControlModifier)
            self.selectShapePoint(pos, multiple_selection_mode=group_mode)
            self.repaint()

    def selectShapePoint(self, point, multiple_selection_mode):
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
        if not self.pixmap:
            return super(PreviewCanvas, self).paintEvent(event)

        p = self._painter
        p.begin(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.setRenderHint(QtGui.QPainter.HighQualityAntialiasing)
        p.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)

        p.scale(self.scale, self.scale)
        p.translate(self.offsetToCenter())
        p.drawPixmap(self.imagePos, self.pixmap)
        # Translate by image position for all shapes
        # so that they move along
        p.translate(self.imagePos)
        Shape.scale = self.scale
        for shape in self.shapes:
            shape.paint(p, fill=(shape in self.selected_shapes or shape == self.hShape))
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
        return QtCore.QPointF(x, y)

    # These two, along with a call to adjustSize are required for the
    # scroll area.
    def sizeHint(self):
        return self.minimumSizeHint()

    def minimumSizeHint(self):
        if self.pixmap:
            return self.scale * self.pixmap.size()
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

    def loadPixmap(self, pixmap):
        self.pixmap = pixmap
        self.shapes = []
        self.repaint()

    def loadShapes(self, shapes, replace=True):
        if replace:
            self.shapes = list(shapes)
        else:
            self.shapes.extend(shapes)
        self.storeShapes()
        self.repaint()
