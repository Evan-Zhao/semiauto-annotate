from PyQt5 import QtCore, QtWidgets
import labelme.utils
from labelme.custom_widgets.preview_canvas import PreviewCanvas
from labelme.custom_widgets.custom_label_qlist import CustomLabelQList


class UIJoinShapesDialog(object):
    def setupUi(self, dialog):
        dialog.setObjectName("Dialog")
        dialog.resize(400, 300)
        dialog.setModal(True)
        mainwindow = dialog.parent().parent()
        self.gridLayout = QtWidgets.QGridLayout(dialog)
        self.gridLayout.setObjectName("gridLayout")
        self.canvas = PreviewCanvas.from_canvas(mainwindow.canvas)
        self.canvas.setObjectName("canvas")
        self.label_list = CustomLabelQList.merge_construct(mainwindow.labelList, self.canvas)
        self.label_list.setObjectName("labelList")
        self.buttonBox = QtWidgets.QDialogButtonBox(dialog)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel | QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.gridLayout.addWidget(self.label_list, 0, 1, 1, 1)
        self.gridLayout.addWidget(self.canvas, 0, 2, 1, 1)
        self.gridLayout.addWidget(self.buttonBox, 1, 0, 1, 2)

        self.retranslateUi(dialog)
        self.buttonBox.accepted.connect(dialog.accept)
        self.buttonBox.rejected.connect(dialog.reject)
        QtCore.QMetaObject.connectSlotsByName(dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Join shapes"))


class JoinShapesDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super(JoinShapesDialog, self).__init__(parent)
        self.ui = UIJoinShapesDialog()
        self.ui.setupUi(self)
        bb = self.ui.buttonBox
        bb.button(bb.Ok).setIcon(labelme.utils.newIcon('done'))
        bb.button(bb.Cancel).setIcon(labelme.utils.newIcon('undo'))
        bb.accepted.connect(self.accepted_close)
        bb.rejected.connect(self.rejected_close)

    def accepted_close(self):
        self.done(0)

    def rejected_close(self):
        self.accepted_close()
