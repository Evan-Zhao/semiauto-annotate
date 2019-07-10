from qtpy import QtCore, QtWidgets
import labelme.utils
from labelme.custom_widgets.preview_canvas import PreviewCanvas
from labelme.custom_widgets.custom_label_qlist import CustomLabelQList


class UIJoinShapesDialog(object):
    def setupUi(self, dialog):
        dialog.setObjectName("Dialog")
        dialog.resize(400, 300)
        dialog.setModal(True)
        self.gridLayout = QtWidgets.QGridLayout(dialog)
        self.gridLayout.setObjectName("gridLayout")
        self.canvas = PreviewCanvas.from_canvas(dialog.parent())
        self.canvas.setObjectName("canvas")
        self.label_list = CustomLabelQList.from_canvas(self.canvas)
        self.label_list.setObjectName("labelList")
        self.buttonBox = QtWidgets.QDialogButtonBox(dialog)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel | QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.buttonBox.button(self.buttonBox.Ok).setIcon(labelme.utils.newIcon('done'))
        self.buttonBox.button(self.buttonBox.Cancel).setIcon(labelme.utils.newIcon('undo'))
        self.gridLayout.addWidget(self.label_list, 0, 1, 1, 1)
        self.gridLayout.addWidget(self.canvas, 0, 2, 1, 1)
        self.gridLayout.addWidget(self.buttonBox, 1, 0, 1, 2)

        self.retranslateUi(dialog)
        self.buttonBox.accepted.connect(dialog.accepted_close)
        self.buttonBox.rejected.connect(dialog.reject)
        QtCore.QMetaObject.connectSlotsByName(dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Join shapes"))


class JoinShapesDialog(QtWidgets.QDialog):
    shapes_joined = QtCore.Signal(list)

    def __init__(self, parent_canvas):
        super(JoinShapesDialog, self).__init__(parent_canvas)
        self.ui = UIJoinShapesDialog()
        self.ui.setupUi(self)

    def accepted_close(self):
        self.shapes_joined.emit(self.ui.label_list.selected_shapes)
        self.accept()
