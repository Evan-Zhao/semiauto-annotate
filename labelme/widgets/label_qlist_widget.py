from qtpy import QtWidgets, QtCore


class LabelQListWidget(QtWidgets.QListWidget):

    def __init__(self, *args, **kwargs):
        super(LabelQListWidget, self).__init__(*args, **kwargs)
        self.canvas = None
        self.itemsToShapes = []
        self.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)

    def get_shape_from_item(self, item):
        for index, (item_, shape) in enumerate(self.itemsToShapes):
            if item_ is item:
                return shape

    def get_item_from_shape(self, shape):
        for index, (item, shape_) in enumerate(self.itemsToShapes):
            if shape_ is shape:
                return item

    def clear(self):
        super(LabelQListWidget, self).clear()
        self.itemsToShapes = []

    def setParent(self, parent):
        self.parent = parent

    def dropEvent(self, event):
        shapes = self.shapes
        super(LabelQListWidget, self).dropEvent(event)
        if self.shapes == shapes:
            return
        if self.canvas is None:
            raise RuntimeError('self.canvas must be set beforehand.')
        self.parent.setDirty()
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

    @property
    def shapes(self):
        shapes = []
        for i in range(self.count()):
            item = self.item(i)
            shape = self.get_shape_from_item(item)
            shapes.append(shape)
        return shapes
