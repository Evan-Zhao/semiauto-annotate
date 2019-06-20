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

    def __init__(self, text="Enter object label", parent=None, labels=None, label_flags=None,
                 sort_labels=True, show_text_field=True,
                 completion='startswith', fit_to_content=None):
        super(LabelDialog, self).__init__(parent)
        self._placeholder = text
        self._show_text_field = show_text_field
        self._sort_labels = sort_labels
        self._labels = labels
        self._label_flags = {} if label_flags is None else label_flags
        self._boxes = []
        self._collect_actions = {}
        self.response = {}
        if fit_to_content is None:
            fit_to_content = {'row': False, 'column': True}
        self._fit_to_content = fit_to_content
        self.completion = completion
        if not QT5 and self.completion != 'startswith':
            logger.warn(
                "completion other than 'startswith' is only "
                "supported with Qt5. Using 'startswith'"
            )
            self.completion = 'startswith'

        # Dialog layout
        layout = QtWidgets.QVBoxLayout()
        self.splitter = QtWidgets.QSplitter()
        layout.addWidget(self.splitter)
        self.setLayout(layout)
        # Leftmost groupbox
        topLevelGroupBox = self.make_group_box(labels, self._collect_actions)
        self.splitter.addWidget(topLevelGroupBox)
        # buttons
        bb = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
            QtCore.Qt.Horizontal,
            self,
        )
        bb.button(bb.Ok).setIcon(labelme.utils.newIcon('done'))
        bb.button(bb.Cancel).setIcon(labelme.utils.newIcon('undo'))
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        layout.addWidget(bb)

    def make_group_box(self, labels, collect_actions):
        label_items = list(labels.keys())
        if '__flags' in labels:
            flags = labels['__flags']
            label_items.remove('__flags')
        else:
            flags = []
        # Group box (parent)
        group_box = QtWidgets.QGroupBox(self)
        verticalLayout = QtWidgets.QVBoxLayout()
        group_box.setLayout(verticalLayout)
        # Edit box
        edit = LabelQLineEdit(group_box)
        edit.setPlaceholderText(self._placeholder)
        edit.setValidator(labelme.utils.labelValidator())
        edit.editingFinished.connect(self.post_process)
        if self._show_text_field:
            verticalLayout.addWidget(edit)
        # Label list
        label_list = QtWidgets.QListWidget(group_box)
        LabelDialog.set_scroll_bar(label_list, self._fit_to_content)
        label_list.addItems(label_items)
        if self._sort_labels:
            label_list.sortItems()
        else:
            label_list.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)
        label_list.currentItemChanged.connect(self.labelSelected)
        edit.setListWidget(label_list)
        verticalLayout.addWidget(label_list)
        collect_actions['__label'] = \
            lambda label_list=label_list: LabelDialog.get_label_list_selected(label_list)
        # completion
        completer = QtWidgets.QCompleter()
        if self.completion == 'startswith':
            completer.setCompletionMode(QtWidgets.QCompleter.InlineCompletion)
            # Default settings.
            # completer.setFilterMode(QtCore.Qt.MatchStartsWith)
        elif self.completion == 'contains':
            completer.setCompletionMode(QtWidgets.QCompleter.PopupCompletion)
            completer.setFilterMode(QtCore.Qt.MatchContains)
        else:
            raise ValueError('Unsupported completion: {}'.format(self.completion))
        completer.setModel(label_list.model())
        # Label flags
        flagsBox = QtWidgets.QGroupBox(group_box)
        flagsLayout = QtWidgets.QFormLayout()
        flagsBox.setLayout(flagsLayout)
        actions = self.set_flags_on(flagsBox, flags)
        collect_actions['__flags'] = actions
        group_box.layout().addWidget(flagsBox)
        # self.edit.textChanged.connect(self.updateFlags)
        # Special label flags
        special_flags_box = QtWidgets.QGroupBox(group_box)
        flagsLayout = QtWidgets.QFormLayout()
        special_flags_box.setLayout(flagsLayout)
        group_box.layout().addWidget(special_flags_box)
        # Add self to list
        self._boxes.append(group_box)
        # Adjust dialog size
        if self._fit_to_content['column']:
            total_width = sum(
                LabelDialog.get_min_width_for_box(box) for box in self._boxes
            )
            self.setMinimumWidth(total_width)
        if self._fit_to_content['row']:
            max_height = max(
                LabelDialog.get_min_height_for_box(box) for box in self._boxes
            )
            self.setMinimumHeight(max_height)
        return group_box

    @staticmethod
    def get_edit_from_box(box):
        return box.layout().itemAt(0).widget()

    @staticmethod
    def get_list_from_box(box):
        return box.layout().itemAt(1).widget()

    @staticmethod
    def get_special_flags_from_box(box):
        return box.layout().itemAt(3).widget()

    @staticmethod
    def get_min_width_for_box(box):
        label_list = LabelDialog.get_list_from_box(box)
        return label_list.sizeHintForColumn(0) * 1.5

    @staticmethod
    def get_min_height_for_box(box):
        label_list = LabelDialog.get_list_from_box(box)
        return label_list.sizeHintForRow(0) * label_list.count() * 1.2

    def find_dicts(self, box):
        def walk_dict(d):
            current = d
            for box in self._boxes[:idx]:
                label_list = LabelDialog.get_list_from_box(box)
                level_i_selected = LabelDialog.get_label_list_selected(label_list)
                current = current[level_i_selected]
            return current

        idx = self._boxes.index(box)
        return idx, walk_dict(self._labels), walk_dict(self._collect_actions)

    @staticmethod
    def get_label_list_selected(label_list):
        # Single selection boxes, only one can be selected
        selected = label_list.selectedItems()
        return selected[0].text() if selected else None

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
        return
        if self.labelList.findItems(label, QtCore.Qt.MatchExactly):
            return
        self.labelList.addItem(label)
        if self._sort_labels:
            self.labelList.sortItems()

    def remove_last_label_effect(self, level, actions, text):
        def clear_layout(layout):
            for i in range(layout.count()):
                layout.itemAt(i).widget().deleteLater()

        actions[text] = None
        actions['__special_flags'] = None
        special_flags_box = self.get_special_flags_from_box(self._boxes[level])
        clear_layout(special_flags_box.layout())
        for box in self._boxes[level + 1:]:
            box.deleteLater()
        self._boxes = self._boxes[:level + 1]

    def labelSelected(self, item):
        label_list = item.listWidget()
        box = label_list.parent()
        edit = LabelDialog.get_edit_from_box(box)
        edit.setText(item.text())

        idx, labels, actions = self.find_dicts(box)
        self.remove_last_label_effect(idx, actions, item.text())

        spec = labels[item.text()]
        if type(spec) is list:
            special_flags_box = self.get_special_flags_from_box(box)
            flag_actions = self.set_flags_on(special_flags_box, spec)
            actions['__special_flags'] = flag_actions
        elif type(spec) is dict:
            next_level_actions = {}
            next_level = self.make_group_box(spec, next_level_actions)
            self._collect_actions[item.text()] = next_level_actions
            self.splitter.addWidget(next_level)

    def post_process(self):
        edit = self.sender()
        text = edit.text()
        if hasattr(text, 'strip'):
            text = text.strip()
        else:
            text = text.trimmed()
        edit.setText(text)

    def set_flags_on(self, widget, flags):
        layout = widget.layout()
        actions = {}
        for v in flags:
            spec = self._label_flags[v]
            if spec == 'bool':
                item = QtWidgets.QCheckBox(v, widget)
                layout.addRow(item)
                actions[v] = lambda item=item: item.isChecked()
                item.show()
            elif spec == 'int':
                label = QtWidgets.QLabel(v, widget)
                edit = QtWidgets.QLineEdit(widget)
                layout.addRow(label, edit)
                actions[v] = lambda edit=edit: edit.text()
                label.show()
                edit.show()
            elif type(spec) == list:
                label = QtWidgets.QLabel(v, widget)
                combo = QtWidgets.QComboBox(widget)
                combo.addItems(spec)
                layout.addRow(label, combo)
                actions[v] = lambda combo=combo: combo.currentText()
                label.show()
                combo.show()
            else:
                assert False
        return actions

    @staticmethod
    def collect_state_recursive(action):
        response = {}
        if '__flags' in action:
            response['__flags'] = {}
            for flag, act in action['__flags'].items():
                response['__flags'][flag] = act()
        label = action['__label']()
        next_action = action.get(label, None)
        response['__label'] = label
        response[label] = None if next_action is None else LabelDialog.collect_state_recursive(next_action)
        return response

    def collect_current_state(self):
        return LabelDialog.collect_state_recursive(self._collect_actions)

    def popUp(self, text=None, move=True, flags=None):
        def validate_result(self):
            return True
            # TODO
            # text = self.edit.text()
            # if hasattr(text, 'strip'):
            #     text = text.strip()
            # else:
            #     text = text.trimmed()
            # if text:

        # if text is None, the previous label in self.edit is kept
        # if text is None:
        #     text = self.edit.text()
        # if flags:
        #     self.setFlags(flags)
        # else:
        #     self.resetFlags(text)
        # self.edit.setText(text)
        # self.edit.setSelection(0, len(text))
        # items = self.labelList.findItems(text, QtCore.Qt.MatchFixedString)
        # if items:
        #     if len(items) != 1:
        #         logger.warning("Label list has duplicate '{}'".format(text))
        #     self.labelList.setCurrentItem(items[0])
        #     row = self.labelList.row(items[0])
        #     self.edit.completer().setCurrentRow(row)
        # self.edit.setFocus(QtCore.Qt.PopupFocusReason)
        if move:
            self.move(QtGui.QCursor.pos())
        if self.exec_():
            result = self.collect_current_state()
            if validate_result(result):
                text = list(set(result.keys()) - {'__flags'})[0]
                return text, result
        return None, None
