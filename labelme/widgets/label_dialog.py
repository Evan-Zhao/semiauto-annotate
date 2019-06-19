import re

from qtpy import QT_VERSION
from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets

QT5 = QT_VERSION[0] == '5'  # NOQA

from labelme.logger import logger
import labelme.utils


# TODO(unknown):
# - Calculate optimal position so as not to go out of screen area.


class LabelQLineEdit(QtWidgets.QLineEdit):

    def setListWidget(self, list_widget):
        self.list_widget = list_widget

    def keyPressEvent(self, e):
        if e.key() in [QtCore.Qt.Key_Up, QtCore.Qt.Key_Down]:
            self.list_widget.keyPressEvent(e)
        else:
            super(LabelQLineEdit, self).keyPressEvent(e)


class LabelDialog(QtWidgets.QDialog):

    def __init__(self, text="Enter object label", parent=None, labels=None,
                 sort_labels=True, show_text_field=True,
                 completion='startswith', fit_to_content=None, flags=None):
        if fit_to_content is None:
            fit_to_content = {'row': False, 'column': True}
        self._fit_to_content = fit_to_content

        super(LabelDialog, self).__init__(parent)

        # Dialog layout
        layout = QtWidgets.QVBoxLayout()
        splitter = QtWidgets.QSplitter()
        layout.addWidget(splitter)
        self.setLayout(layout)
        # Left side layout
        leftGroup = QtWidgets.QGroupBox()
        leftLayout = QtWidgets.QVBoxLayout()
        leftGroup.setLayout(leftLayout)
        splitter.addWidget(leftGroup)
        # Right side layout
        self.rightStack = QtWidgets.QStackedWidget()
        splitter.addWidget(self.rightStack)

        # Edit box
        self.edit = LabelQLineEdit()
        self.edit.setPlaceholderText(text)
        self.edit.setValidator(labelme.utils.labelValidator())
        self.edit.editingFinished.connect(self.postProcess)
        if flags:
            self.edit.textChanged.connect(self.updateFlags)
        if show_text_field:
            leftLayout.addWidget(self.edit)
        # buttons
        self.buttonBox = bb = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
            QtCore.Qt.Horizontal,
            self,
        )
        bb.button(bb.Ok).setIcon(labelme.utils.newIcon('done'))
        bb.button(bb.Cancel).setIcon(labelme.utils.newIcon('undo'))
        bb.accepted.connect(self.validate)
        bb.rejected.connect(self.reject)
        leftLayout.addWidget(bb)
        # label_list
        self.labelList = QtWidgets.QListWidget()
        LabelDialog.set_scroll_bar(self.labelList, self._fit_to_content)
        self._sort_labels = sort_labels
        self._labels = labels
        if labels:
            self.labelList.addItems(labels.keys())
        if self._sort_labels:
            self.labelList.sortItems()
        else:
            self.labelList.setDragDropMode(
                QtWidgets.QAbstractItemView.InternalMove)
        self.labelList.currentItemChanged.connect(self.labelSelected)
        self.edit.setListWidget(self.labelList)
        leftLayout.addWidget(self.labelList)
        # label_flags
        self._flags = {} if flags is None else flags
        flagsBox = QtWidgets.QGroupBox()
        self.flagsLayout = QtWidgets.QVBoxLayout()
        flagsBox.setLayout(self.flagsLayout)
        self.resetFlags()
        leftLayout.addWidget(flagsBox)
        self.edit.textChanged.connect(self.updateFlags)
        # sublabel / form (for lane)
        blankPage = QtWidgets.QGroupBox(self.rightStack)
        self.sublabelList = QtWidgets.QListWidget(self.rightStack)
        self.laneForm = QtWidgets.QGroupBox(self.rightStack)
        self.laneFormLayout = QtWidgets.QFormLayout()
        self.laneForm.setLayout(self.laneFormLayout)
        self.rightStack.addWidget(blankPage)
        self.rightStack.addWidget(self.sublabelList)
        self.rightStack.addWidget(self.laneForm)

        # completion
        completer = QtWidgets.QCompleter()
        if not QT5 and completion != 'startswith':
            logger.warn(
                "completion other than 'startswith' is only "
                "supported with Qt5. Using 'startswith'"
            )
            completion = 'startswith'
        if completion == 'startswith':
            completer.setCompletionMode(QtWidgets.QCompleter.InlineCompletion)
            # Default settings.
            # completer.setFilterMode(QtCore.Qt.MatchStartsWith)
        elif completion == 'contains':
            completer.setCompletionMode(QtWidgets.QCompleter.PopupCompletion)
            completer.setFilterMode(QtCore.Qt.MatchContains)
        else:
            raise ValueError('Unsupported completion: {}'.format(completion))
        completer.setModel(self.labelList.model())
        self.edit.setCompleter(completer)

    @staticmethod
    def set_scroll_bar(component, fit_to_content):
        if fit_to_content['row']:
            component.setHorizontalScrollBarPolicy(
                QtCore.Qt.ScrollBarAlwaysOff
            )
        if fit_to_content['column']:
            component.setVerticalScrollBarPolicy(
                QtCore.Qt.ScrollBarAlwaysOff
            )

    def addLabelHistory(self, label):
        if self.labelList.findItems(label, QtCore.Qt.MatchExactly):
            return
        self.labelList.addItem(label)
        if self._sort_labels:
            self.labelList.sortItems()

    @staticmethod
    def clearLayout(layout):
        for i in range(layout.count()):
            layout.itemAt(i).widget().deleteLater()

    def labelSelected(self, item):
        self.edit.setText(item.text())
        # Set to blank page by default
        self.rightStack.setCurrentIndex(0)
        if item.text() not in self._labels:
            return
        spec = self._labels[item.text()]
        if type(spec) is list:
            # Set to page 0
            self.rightStack.setCurrentIndex(1)
            self.sublabelList.clear()
            self.sublabelList.addItems(spec)
        elif type(spec) is dict:
            # Set to page 1
            self.rightStack.setCurrentIndex(2)
            LabelDialog.clearLayout(self.laneFormLayout)
            for key, values in spec.items():
                label = QtWidgets.QLabel(key, self)
                combo = QtWidgets.QComboBox(self)
                combo.addItems(values)
                label.show()
                combo.show()
                self.laneFormLayout.addRow(label, combo)

    def validate(self):
        text = self.edit.text()
        if hasattr(text, 'strip'):
            text = text.strip()
        else:
            text = text.trimmed()
        if text:
            self.accept()

    def postProcess(self):
        text = self.edit.text()
        if hasattr(text, 'strip'):
            text = text.strip()
        else:
            text = text.trimmed()
        self.edit.setText(text)

    def updateFlags(self, label_new):
        # keep state of shared flags
        flags_old = self.getFlags()

        flags_new = {}
        for pattern, keys in self._flags.items():
            if re.match(pattern, label_new):
                for key in keys:
                    flags_new[key] = flags_old.get(key, False)
        self.setFlags(flags_new)

    def deleteFlags(self):
        for i in reversed(range(self.flagsLayout.count())):
            item = self.flagsLayout.itemAt(i).widget()
            self.flagsLayout.removeWidget(item)
            item.setParent(None)

    def resetFlags(self, label=''):
        flags = {}
        for pattern, keys in self._flags.items():
            if re.match(pattern, label):
                for key in keys:
                    flags[key] = False
        self.setFlags(flags)

    def setFlags(self, flags):
        self.deleteFlags()
        for key in flags:
            item = QtWidgets.QCheckBox(key, self)
            item.setChecked(flags[key])
            self.flagsLayout.addWidget(item)
            item.show()

    def getFlags(self):
        flags = {}
        for i in range(self.flagsLayout.count()):
            item = self.flagsLayout.itemAt(i).widget()
            flags[item.text()] = item.isChecked()
        return flags

    def collectCurrentState(self):
        def collectFormDict(layout):
            ret = {}
            while layout.rowCount() > 0:
                row = layout.takeRow(0)
                label, field = row.labelItem.widget(), row.fieldItem.widget(),
                ret[label.text()] = field.currentText()
            return ret

        idx = self.rightStack.currentIndex()
        if idx == 0:
            extended = None
        elif idx == 1:
            extended = self.sublabelList.currentItem().text()
        else:
            assert idx == 2
            extended = collectFormDict(self.laneFormLayout)
        return self.edit.text(), self.getFlags(), extended

    def popUp(self, text=None, move=True, flags=None):
        if self._fit_to_content['row']:
            self.labelList.setMinimumHeight(
                self.labelList.sizeHintForRow(0) * self.labelList.count() + 2
            )
        if self._fit_to_content['column']:
            self.labelList.setMinimumWidth(
                self.labelList.sizeHintForColumn(0) + 2
            )
        # if text is None, the previous label in self.edit is kept
        if text is None:
            text = self.edit.text()
        if flags:
            self.setFlags(flags)
        else:
            self.resetFlags(text)
        self.edit.setText(text)
        self.edit.setSelection(0, len(text))
        items = self.labelList.findItems(text, QtCore.Qt.MatchFixedString)
        if items:
            if len(items) != 1:
                logger.warning("Label list has duplicate '{}'".format(text))
            self.labelList.setCurrentItem(items[0])
            row = self.labelList.row(items[0])
            self.edit.completer().setCurrentRow(row)
        self.edit.setFocus(QtCore.Qt.PopupFocusReason)
        if move:
            self.move(QtGui.QCursor.pos())
        if self.exec_():
            return self.collectCurrentState()
        else:
            return None, None, None
