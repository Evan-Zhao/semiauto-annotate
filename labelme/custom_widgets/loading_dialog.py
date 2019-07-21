from qtpy.QtCore import QMetaObject, Signal, Qt
from qtpy.QtWidgets import QDialog, QHBoxLayout, QLabel
from qtpy.QtGui import QFont

from .waiting_spinner import QtWaitingSpinner


class UILoadingDialog(object):
    def __init__(self):
        self.layout = self.waiting = self.label = None

    def setup_ui(self, dialog):
        dialog.setObjectName("Dialog")
        dialog.resize(600, 200)
        dialog.setModal(True)
        self.layout = QHBoxLayout(dialog)
        self.layout.setObjectName("hbox_layout")
        self.waiting = QtWaitingSpinner(dialog, centerOnParent=False)
        self.waiting.setObjectName("waiting")
        self.label = QLabel('Loading model result...')
        font = self.label.font()
        font.setPointSize(16)
        self.label.setFont(font)
        self.label.setStyleSheet("QLabel { color : black; }")
        self.layout.addWidget(self.waiting)
        self.layout.addWidget(self.label)
        QMetaObject.connectSlotsByName(dialog)


class LoadingDialog(QDialog):
    loaded = Signal(bool)

    def __init__(self, parent):
        super(LoadingDialog, self).__init__(
            parent, Qt.SplashScreen
        )
        self.ui = UILoadingDialog()
        self.ui.setup_ui(self)
        self.loaded.connect(self.accept)

    def on_loaded(self, succeeded):
        if succeeded:
            self.accept()
        else:
            print('呀 嘿 嘿 啊啊啊啊啊啊啊')
            self.accept()

    def exec_(self):
        self.ui.waiting.start()
        super().exec_()

    def accept(self):
        self.ui.waiting.stop()
        super().accept()
