from qtpy import QtWidgets, QtCore


class CustomLabelQList(QtWidgets.QListWidget):
    checked_item_changed = QtCore.Signal(list)

    def __init__(self, canvas, *args, **kwargs):
        super(CustomLabelQList, self).__init__(*args, **kwargs)
        # `canvas` is a PreviewCanvas
        self.canvas = canvas
        self.items_to_shapes = []
        self._anti_signal_loop = False
        self.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        # Check the checkboxes of all selected items.
        self.itemSelectionChanged.connect(self.check_selected_item)
        # Bidirectional signal binding;
        # Click item -> select shape; click shape -> select item
        # 1. We bind ourselves itemChanged to emit checked_item_changed
        # 2. Canvas binds to our checked_item_changed (when checkbox changes)
        # 3. We bind to canvas selectionChanged
        self.itemChanged.connect(self.on_item_changed)
        canvas.selectionChanged.connect(self.on_selection_change)
        canvas.set_selection_signal(self.checked_item_changed)

    def check_selected_item(self):
        """Check the checkbox of selected item on selection change."""
        for item in self.list_items:
            item.setCheckState(QtCore.Qt.Unchecked)
        for item in self.selectedItems():
            item.setCheckState(QtCore.Qt.Checked)

    def on_item_changed(self):
        # If self._anti_signal_loop is True, this change is
        # due to inbound signal from Canvas; nothing should be emitted,
        # in order to prevent signal loop.
        if self._anti_signal_loop:
            return
        self.checked_item_changed.emit(self.selected_shapes)

    @property
    def selected_shapes(self):
        selected_shapes = [
            self.get_shape_from_item(item)
            for item in self.list_items if item.checkState() == QtCore.Qt.Checked
        ]
        return selected_shapes

    def on_selection_change(self):
        # Prevent signal loop.
        self._anti_signal_loop = True
        self.deselect_and_uncheck_all()
        for shape in self.canvas.selected_shapes:
            item = self.get_item_from_shape(shape)
            item.setSelected(True)
            item.setCheckState(QtCore.Qt.Checked)
        self._anti_signal_loop = False

    @classmethod
    def from_canvas(cls, canvas, *args, **kwargs):
        """Construct from a Canvas (or PreviewCanvas)."""
        this = cls(canvas, *args, **kwargs)
        for shape in canvas.shapes:
            item = QtWidgets.QListWidgetItem(shape.label)
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.Unchecked)
            this.addItem(item)
            this.items_to_shapes.append((item, shape))
        return this

    def get_shape_from_item(self, item):
        for index, (item_, shape) in enumerate(self.items_to_shapes):
            if item_ is item:
                return shape

    def get_item_from_shape(self, shape):
        for index, (item, shape_) in enumerate(self.items_to_shapes):
            if shape_ is shape:
                return item

    def clear(self):
        super(CustomLabelQList, self).clear()
        self.items_to_shapes = []

    def dropEvent(self, event):
        shapes = self.shapes
        super(CustomLabelQList, self).dropEvent(event)
        if self.shapes == shapes:
            return
        self.parent().setDirty()
        self.canvas.loadShapes(self.shapes)

    def invertSelection(self):
        for i in range(self.count()):
            listItem = self.item(i)
            if listItem.checkState() == QtCore.Qt.Checked:
                listItem.setCheckState(QtCore.Qt.Unchecked)
                listItem.setSelected(False)
            elif listItem.checkState() == QtCore.Qt.Unchecked:
                listItem.setCheckState(QtCore.Qt.Checked)
                listItem.setSelected(True)
            else:
                assert False

    def select_and_check_all(self):
        self.selectAll()
        for i in range(self.count()):
            listItem = self.item(i)
            listItem.setCheckState(QtCore.Qt.Checked)

    def deselect_and_uncheck_all(self):
        self.clearSelection()
        for i in range(self.count()):
            listItem = self.item(i)
            listItem.setCheckState(QtCore.Qt.Unchecked)

    @property
    def shapes(self):
        shapes = []
        for i in range(self.count()):
            item = self.item(i)
            shape = self.get_shape_from_item(item)
            shapes.append(shape)
        return shapes

    @property
    def list_items(self):
        return [self.item(i) for i in range(self.count())]
