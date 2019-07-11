from qtpy import QtCore, QtWidgets
import labelme.utils
from labelme.custom_widgets.preview_canvas import PreviewCanvas
from labelme.utils import Config


class UIPoseAnnotationDialog(object):
    def setupUi(self, dialog):
        dialog.setObjectName("Dialog")
        dialog.resize(400, 300)
        dialog.setModal(True)
        self.gridLayout = QtWidgets.QGridLayout(dialog)
        self.gridLayout.setObjectName("gridLayout")
        self.canvas = PreviewCanvas.from_canvas(dialog.parent())
        self.canvas.setObjectName("canvas")
        self.point_labels_list = QtWidgets.QListWidget(dialog)
        self.point_labels_list.setObjectName("point_labels_list")
        self.buttonBox = QtWidgets.QDialogButtonBox(dialog)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel | QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.buttonBox.button(self.buttonBox.Ok).setIcon(labelme.utils.newIcon('done'))
        self.buttonBox.button(self.buttonBox.Cancel).setIcon(labelme.utils.newIcon('undo'))
        self.gridLayout.addWidget(self.point_labels_list, 0, 1, 1, 1)
        self.gridLayout.addWidget(self.canvas, 0, 2, 1, 1)
        self.gridLayout.addWidget(self.buttonBox, 1, 0, 1, 2)

        self.buttonBox.accepted.connect(dialog.accept)
        self.buttonBox.rejected.connect(dialog.reject)
        QtCore.QMetaObject.connectSlotsByName(dialog)


class PoseAnnotationDialog(QtWidgets.QDialog):
    def __init__(self, parent_canvas, selected_shape):
        super(PoseAnnotationDialog, self).__init__(parent_canvas)
        self.ui = UIPoseAnnotationDialog()
        self.ui.setupUi(self)
        self._point_labels = Config.get('point_labels')
