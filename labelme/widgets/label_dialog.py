from qtpy import QT_VERSION
from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets

QT5 = QT_VERSION[0] == '5'  # NOQA

from labelme.logger import logger
from labelme.custom_widgets.join_shapes_dialog import JoinShapesDialog
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


class DialogContinuation(object):
    """"""

    def __init__(self, parent_dialog):
        self.parent_dialog = parent_dialog
        self.child_dialog = None

    @staticmethod
    def void(arg):
        pass

    def exec_(self, child_ret_processor=None):
        v = DialogContinuation.void
        if child_ret_processor is None:
            child_ret_processor = v
        parent_ret = self.parent_dialog.exec_()
        while self.child_dialog:
            child_ret = self.child_dialog.exec_()
            child_ret_processor(child_ret)
            self.child_dialog = None
            parent_ret = self.parent_dialog.exec_()
        return parent_ret

    def switch_to_child(self, child_dialog):
        # Hide parent dialog. Call to exec_() will return
        # Store child dialog and execute it in our exec_()
        # to prevent it from returning
        self.parent_dialog.hide()
        self.child_dialog = child_dialog


class LabelDialog(QtWidgets.QDialog):
    def __init__(self, text="Enter object label", parent=None, labels=None, label_flags=None,
                 sort_labels=True, show_text_field=True,
                 completion='startswith', fit_to_content=None):
        super(LabelDialog, self).__init__(parent)
        self._form = None
        self._placeholder = text
        self._show_text_field = show_text_field
        self._sort_labels = sort_labels
        self._labels = labels
        self._label_flags = {} if label_flags is None else label_flags
        self._boxes = []
        self._bindings = {}
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
        top_level_group_box = self.make_group_box(labels, self._bindings)
        self.add_box(top_level_group_box)
        # buttons
        bb = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
            QtCore.Qt.Horizontal,
            self,
        )
        bb.button(bb.Ok).setIcon(labelme.utils.newIcon('done'))
        bb.button(bb.Cancel).setIcon(labelme.utils.newIcon('undo'))
        bb.accepted.connect(self.validate)
        bb.rejected.connect(self.reject)
        layout.addWidget(bb)

        self.join_dialog = None
        self.dialog_continuation = DialogContinuation(self)

    @property
    def edit(self):
        return self.get_edit_from_box(self._boxes[0])

    def make_group_box(self, labels, bindings):
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
        bindings['__label'] = (
            lambda lst=label_list: LabelDialog.get_label_list_selected(lst),
            lambda val, lst=label_list: self.set_label_list_selected(lst, val)
        )
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
        edit.setCompleter(completer)
        # Label flags
        flags_box = QtWidgets.QGroupBox(group_box)
        flags_layout = QtWidgets.QFormLayout()
        flags_box.setLayout(flags_layout)
        flags_box.setVisible(False)
        bindings['__flags'] = self.make_flags_on(flags_box, flags)
        group_box.layout().addWidget(flags_box)
        # Special label flags
        special_flags_box = QtWidgets.QGroupBox(group_box)
        flags_layout = QtWidgets.QFormLayout()
        special_flags_box.setLayout(flags_layout)
        group_box.layout().addWidget(special_flags_box)
        special_flags_box.setVisible(False)
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
            for b in self._boxes[:idx]:
                label_list = LabelDialog.get_list_from_box(b)
                level_i_selected = LabelDialog.get_label_list_selected(label_list)
                current = current[level_i_selected]
            return current

        idx = self._boxes.index(box)
        return idx, walk_dict(self._labels), walk_dict(self._bindings)

    @staticmethod
    def get_label_list_selected(label_list):
        # Single selection boxes, only one can be selected
        selected = label_list.selectedItems()
        if not selected:
            raise ValueError('Select something!')
        return selected[0].text()

    def set_label_list_selected(self, label_list, text):
        items = label_list.findItems(text, QtCore.Qt.MatchFixedString)
        edit = self.get_edit_from_box(label_list.parent())
        if items:
            if len(items) != 1:
                logger.warning("Label list has duplicate '{}'".format(text))
            label_list.setCurrentItem(items[0])
            row = label_list.row(items[0])
            edit.completer().setCurrentRow(row)

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

    def remove_last_label_effect(self, level, bindings, text):
        def clear_layout(layout):
            for i in range(layout.count()):
                layout.itemAt(i).widget().deleteLater()

        bindings[text] = None
        bindings['__special_flags'] = None
        special_flags_box = self.get_special_flags_from_box(self._boxes[level])
        clear_layout(special_flags_box.layout())
        for box in self._boxes[level + 1:]:
            box.deleteLater()
        self._boxes = self._boxes[:level + 1]

    def add_box(self, box):
        self.splitter.addWidget(box)
        self._boxes.append(box)

    def labelSelected(self, item):
        label_list = item.listWidget()
        box = label_list.parent()
        edit = LabelDialog.get_edit_from_box(box)
        edit.setText(item.text())

        idx, labels, bindings = self.find_dicts(box)
        self.remove_last_label_effect(idx, bindings, item.text())

        spec = labels[item.text()]
        if type(spec) is list:
            special_flags_box = self.get_special_flags_from_box(box)
            flag_actions = self.make_flags_on(special_flags_box, spec)
            bindings['__special_flags'] = flag_actions
        elif type(spec) is dict:
            next_bindings = {}
            self._bindings[item.text()] = next_bindings
            next_level = self.make_group_box(spec, next_bindings)
            self.add_box(next_level)
        elif self._label_flags.get(spec, None) == 'canvas':
            # Special case
            group_box = QtWidgets.QGroupBox(self)
            vertical_layout = QtWidgets.QVBoxLayout()
            group_box.setLayout(vertical_layout)
            join_button = QtWidgets.QPushButton(parent=group_box, text='Join more shapes')
            join_button.clicked.connect(self.open_join_dialog)
            key_points_button = QtWidgets.QPushButton(parent=group_box, text='Annotate key points')
            key_points_button.clicked.connect(self.open_key_points_dialog)
            vertical_layout.addWidget(join_button)
            vertical_layout.addWidget(key_points_button)
            self.add_box(group_box)

    def open_join_dialog(self):
        """
        Opens join dialog on button push.

        After join dialog closes, the control doesn't return to this dialog,
        because # of shapes may have changed.
        """
        self.reject()
        self.parent().join_shapes_dialog.exec_()

    def open_key_points_dialog(self):
        # TODO: key points dialog
        pass

    def post_process(self):
        edit = self.sender()
        if type(edit) is not QtWidgets.QLineEdit:
            return
        text = edit.text()
        if hasattr(text, 'strip'):
            text = text.strip()
        else:
            text = text.trimmed()
        edit.setText(text)

    def make_flags_on(self, widget, flags):
        layout = widget.layout()
        bindings = {}
        for v in flags:
            spec = self._label_flags[v]
            if spec == 'bool':
                item = QtWidgets.QCheckBox(v, widget)
                layout.addRow(item)
                bindings[v] = (
                    lambda i=item: i.isChecked(),
                    lambda boolean, i=item: i.setChecked(boolean)
                )
                item.show()
            elif spec == 'int':
                label = QtWidgets.QLabel(v, widget)
                edit = QtWidgets.QLineEdit(widget)
                layout.addRow(label, edit)
                bindings[v] = (
                    lambda e=edit: e.text(),
                    lambda val, e=edit: e.setText(str(val))
                )
                label.show()
                edit.show()
            elif type(spec) == list:
                label = QtWidgets.QLabel(v, widget)
                combo = QtWidgets.QComboBox(widget)
                combo.addItems(spec)
                layout.addRow(label, combo)
                bindings[v] = (
                    lambda c=combo: c.currentText(),
                    lambda val, c=combo: c.setCurrentText(val)
                )
                label.show()
                combo.show()
            else:
                assert False
        if flags:
            widget.setVisible(True)
        return bindings

    @staticmethod
    def collect_state_recursive(bindings):
        response = {}
        if '__flags' in bindings:
            response['__flags'] = {}
            for flag, (getter, _) in bindings['__flags'].items():
                response['__flags'][flag] = getter()
        label = bindings['__label'][0]()
        next_getter = bindings.get(label, None)
        response['__label'] = label
        response[label] = None if next_getter is None else LabelDialog.collect_state_recursive(next_getter)
        return response

    def collect_current_state(self):
        try:
            return LabelDialog.collect_state_recursive(self._bindings)
        except ValueError:  # in case some label list has no selection
            return None

    def set_state_recursive(self, bindings, form):
        flags = form['__flags']
        for flag, (_, setter) in bindings['__flags'].items():
            if flag in flags:
                setter(flags[flag])
        label = form['__label']
        bindings['__label'][1](label)
        next_form = form[label]
        next_bindings = bindings[label]
        if next_form is not None:
            self.set_state_recursive(next_bindings, next_form)

    def validate(self):
        result = self.collect_current_state()
        if result is not None:
            self._form = result
            self.accept()
        else:
            QtWidgets.QMessageBox.warning(self, 'Error', 'Label form is not complete')

    def popUp(self, text_or_form=None, move=True):
        def dummy_form(t):
            return {
                '__flags': {},
                '__label': t,
                t: None
            }

        edit0 = self.get_edit_from_box(self._boxes[0])
        if type(text_or_form) is str:
            text = text_or_form
            self.set_state_recursive(self._bindings, dummy_form(text))
        elif type(text_or_form) is dict:
            form = text_or_form
            self.set_state_recursive(self._bindings, form)
        edit0.setFocus(QtCore.Qt.PopupFocusReason)

        if move:
            self.move(QtGui.QCursor.pos())
        if self.dialog_continuation.exec_():
            assert self._form is not None
            text = self._form['__label']
            return text, self._form
        return None, None
