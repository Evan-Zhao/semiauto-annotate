from qtpy import QtWidgets


class ColorDialog(QtWidgets.QColorDialog):

    def __init__(self, parent=None):
        super(ColorDialog, self).__init__(parent)
        self.setOption(QtWidgets.QColorDialog.ShowAlphaChannel)
        # The Mac native dialog does not support our restore button.
        self.setOption(QtWidgets.QColorDialog.DontUseNativeDialog)
        # Add a restore defaults button.
        # The default is set at invocation time, so that it
        # works across dialogs for different elements.
        self.default = None
        self.return_default = False
        self.bb = self.layout().itemAt(1).widget()
        self.bb.addButton(QtWidgets.QDialogButtonBox.RestoreDefaults)
        self.bb.clicked.connect(self.checkRestore)

    def getColor(self, value=None, title=None, default=None):
        self.default = default
        self.return_default = False
        if title:
            self.setWindowTitle(title)
        if value:
            self.setCurrentColor(value)
        if self.exec_():
            ret_value = self.default if self.return_default else self.currentColor()
            return True, ret_value
        else:
            return False, None

    def checkRestore(self, button):
        if self.bb.buttonRole(button) & QtWidgets.QDialogButtonBox.ResetRole:
            if self.default:
                self.setCurrentColor(self.default)
            self.return_default = True
