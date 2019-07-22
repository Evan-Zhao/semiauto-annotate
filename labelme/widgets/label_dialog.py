from qtpy import QT_VERSION
from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets

QT5 = QT_VERSION[0] == '5'  # NOQA

from labelme.logger import logger
from labelme.custom_widgets import PoseAnnotationWidget
import labelme.utils

from collections import MutableMapping


# TODO(unknown):
# - Calculate optimal position so as not to go out of screen area.


class LabelQLineEdit(QtWidgets.QLineEdit):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.list_widget = None
        self.editingFinished.connect(self.post_process)

    def setListWidget(self, list_widget):
        self.list_widget = list_widget

    def keyPressEvent(self, e):
        if e.key() in [QtCore.Qt.Key_Up, QtCore.Qt.Key_Down]:
            self.list_widget.keyPressEvent(e)
        else:
            super(LabelQLineEdit, self).keyPressEvent(e)

    def post_process(self):
        self.setText(self.text().strip())


class MultiLevelDict(MutableMapping):
    def __init__(self, init_dict):
        self._data = init_dict

    def __getitem__(self, item):
        if type(item) in (list, tuple):
            ret = self._data
            for k in item:
                ret = ret[k]
        else:
            ret = self._data[item]
        if type(ret) is dict:
            return MultiLevelDict(ret)
        else:
            return ret

    def __setitem__(self, key, value):
        raise NotImplementedError()

    def __delitem__(self, key):
        raise NotImplementedError()

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __keytransform__(self, key):
        return key


class UILabelDialog(object):
    _flags = '__flags'
    _placeholder = 'Enter object label'

    def __init__(self, labels, parent_dialog, parent_canvas):
        from labelme.utils import Config

        self.splitter = QtWidgets.QSplitter()
        self.spec = MultiLevelDict(labels)
        self.parent_dialog = parent_dialog
        self.parent_canvas = parent_canvas
        self.boxes = []
        self._using_dialog_box = False
        # Turns off slot response
        self.setting_state = False
        fit_to_content = Config.get('fit_to_content')
        self.fit_row, self.fit_column = False, True
        if fit_to_content:
            self.fit_row, self.fit_column = fit_to_content['row'], fit_to_content['column']

    @staticmethod
    def make_empty_group_box(parent):
        group_box = QtWidgets.QGroupBox(parent)
        v_layout = QtWidgets.QVBoxLayout()
        group_box.setLayout(v_layout)
        return group_box, v_layout

    @staticmethod
    def make_edit_box(parent):
        edit = LabelQLineEdit(parent)
        edit.setPlaceholderText(UILabelDialog._placeholder)
        edit.setValidator(labelme.utils.labelValidator())
        return edit

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

    @staticmethod
    def make_label_list(parent, items, dialog):
        from labelme.utils import Config

        fit_to_content = Config.get('fit_to_content', default={'row': False, 'column': True})
        label_list = QtWidgets.QListWidget(parent)
        UILabelDialog.set_scroll_bar(label_list, fit_to_content)
        label_list.addItems(items)
        if Config.get('sort_labels'):
            label_list.sortItems()
        else:
            label_list.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)
        label_list.currentItemChanged.connect(dialog.label_selected)
        return label_list

    @staticmethod
    def set_label_list_selected(label_list, completer, text):
        items = label_list.findItems(text, QtCore.Qt.MatchFixedString)
        if items:
            if len(items) != 1:
                logger.warning("Label list has duplicate '{}'".format(text))
            label_list.setCurrentItem(items[0])
            row = label_list.row(items[0])
            completer.setCurrentRow(row)

    @staticmethod
    def make_completer(parent):
        from labelme.utils import Config

        completion = Config.get('label_completion')
        completer = QtWidgets.QCompleter(parent)
        if completion == 'startswith':
            completer.setCompletionMode(QtWidgets.QCompleter.InlineCompletion)
            # Default settings.
            # completer.setFilterMode(QtCore.Qt.MatchStartsWith)
        elif completion == 'contains':
            completer.setCompletionMode(QtWidgets.QCompleter.PopupCompletion)
            completer.setFilterMode(QtCore.Qt.MatchContains)
        else:
            raise ValueError('Unsupported completion: {}'.format(completion))
        return completer

    @staticmethod
    def make_empty_flags_box(parent):
        flags_box = QtWidgets.QGroupBox(parent)
        flags_layout = QtWidgets.QFormLayout()
        flags_box.setLayout(flags_layout)
        flags_box.setVisible(False)
        return flags_box

    @staticmethod
    def fill_flags_box(widget, flags, state=None):
        layout = widget.layout()
        for spec in flags:
            kind = UILabelDialog.translate_flag(spec)
            set_value = state.get(spec, None) if state else None
            if kind == 'bool':
                item = QtWidgets.QCheckBox(spec, widget)
                layout.addRow(item)
                if set_value:
                    item.setChecked(set_value)
                item.show()
            elif kind == 'int':
                label = QtWidgets.QLabel(spec, widget)
                edit = QtWidgets.QLineEdit(widget)
                layout.addRow(label, edit)
                if set_value:
                    edit.setText(str(set_value))
                label.show()
                edit.show()
            elif type(kind) == list:
                label = QtWidgets.QLabel(spec, widget)
                combo = QtWidgets.QComboBox(widget)
                combo.addItems(kind)
                layout.addRow(label, combo)
                if set_value:
                    combo.setCurrentText(set_value)
                label.show()
                combo.show()
            else:
                assert False
        if flags:
            widget.setVisible(True)

    @staticmethod
    def assemble_group_box(dialog, labels, flags, set_label=None, set_flag=None):
        from labelme.utils import Config

        # Make widgets
        group_box, v_layout = UILabelDialog.make_empty_group_box(parent=dialog)
        edit = UILabelDialog.make_edit_box(group_box)
        label_list = UILabelDialog.make_label_list(group_box, labels, dialog)
        completer = UILabelDialog.make_completer(group_box)
        UILabelDialog.set_label_list_selected(label_list, completer, set_label)
        special = UILabelDialog.make_empty_flags_box(group_box)
        flags_box = UILabelDialog.make_empty_flags_box(group_box)
        UILabelDialog.fill_flags_box(flags_box, flags, state=set_flag)
        # Set up widget interaction
        edit.setListWidget(label_list)
        edit.setCompleter(completer)
        completer.setModel(label_list.model())
        # Add children widgets to layout
        if Config.get('show_label_text_field'):
            v_layout.addWidget(edit)
        v_layout.addWidget(label_list)
        v_layout.addWidget(flags_box)
        v_layout.addWidget(special)
        return group_box

    def make_pose_group_box(self, dialog, canvas):
        group_box, v_layout = UILabelDialog.make_empty_group_box(parent=self.parent_dialog)
        pose_widget = PoseAnnotationWidget(
            self.parent_dialog, self.parent_canvas, self.parent_dialog._selected_shape
        )
        v_layout.addWidget(pose_widget)
        return group_box

    @staticmethod
    def translate_flag(flag):
        from labelme.utils import Config
        flag_specs = Config.get('label_flags', default={})
        return flag_specs.get(flag, None)

    @staticmethod
    def get_edit_from_box(box):
        return box.layout().itemAt(0).widget()

    @staticmethod
    def get_list_from_box(box):
        return box.layout().itemAt(1).widget()

    @staticmethod
    def get_flags_from_box(box):
        return box.layout().itemAt(2).widget()

    @staticmethod
    def get_special_flags_from_box(box):
        return box.layout().itemAt(3).widget()

    @staticmethod
    def get_keys_from_label_dict(label_dict):
        return list(set(label_dict.keys()) - {UILabelDialog._flags})

    @property
    def group_boxes(self):
        return self.boxes[:-1] if self._using_dialog_box else self.boxes

    def get_texts_selected(self) -> list:
        all_selected = []
        for level in self.group_boxes:
            label_list = self.get_list_from_box(level)
            selected_item = label_list.currentItem()
            if selected_item is None:
                return all_selected
            all_selected.append(selected_item.text())
        return all_selected

    def get_full_state(self):
        def read_flags_box(flags_box, spec):
            def get_widget(row, role):
                return flags_box.layout().itemAt(row, role).widget()

            spanning_role = QtWidgets.QFormLayout.SpanningRole
            field_role = QtWidgets.QFormLayout.FieldRole
            ret = {}
            for i, s in enumerate(spec):
                kind = self.translate_flag(s)
                if kind == 'bool':
                    val = get_widget(i, spanning_role).isChecked()
                elif kind == 'int':
                    val = get_widget(i, field_role).text()
                elif type(kind) == list:
                    val = get_widget(i, field_role).currentText()
                else:
                    assert False
                ret[s] = val
            return ret

        form = []
        all_sel = self.get_texts_selected()
        if len(all_sel) < len(self.group_boxes):
            # Some label list has no selection.
            return None, None
        alt_shapes = None
        for level, box in enumerate(self.group_boxes):
            box = self.boxes[level]
            parent_selected = all_sel[:level]
            level_spec = self.spec[parent_selected]
            flags, special_flags = None, None
            if self._flags in level_spec:
                flags_box = self.get_flags_from_box(box)
                flags_spec = level_spec[self._flags]
                flags = read_flags_box(flags_box, flags_spec)
            level_selected = all_sel[level] if len(all_sel) > level else None
            if level_selected:
                child_spec = level_spec[level_selected]
                if type(child_spec) is list:
                    special_flags_box = self.get_special_flags_from_box(box)
                    special_flags = read_flags_box(special_flags_box, child_spec)
            form.append([level_selected, flags, special_flags])
        if self._using_dialog_box:
            pose_widget = self.boxes[-1].layout().itemAt(0).widget()
            alt_shapes = pose_widget.get_shapes()
        return form, alt_shapes

    @staticmethod
    def get_first_label_from_state(state):
        return state[0][0]

    def set_size(self):
        def get_min_width_for_box(box):
            label_list = self.get_list_from_box(box)
            return label_list.sizeHintForColumn(0) * 2

        def get_min_height_for_box(box):
            total_suggested = box.sizeHint().height()
            label_list = self.get_list_from_box(box)
            label_list_suggested = label_list.sizeHint().height()
            label_list_actual = label_list.sizeHintForRow(0) * label_list.count() * 1.5
            return total_suggested - label_list_suggested + label_list_actual

        if self.fit_row:
            total_width = sum(
                get_min_width_for_box(box) for box in self.group_boxes
            )
            if self._using_dialog_box:
                total_width += 100
            self.parent_dialog.setMinimumWidth(total_width)
        if self.fit_column:
            heights = [get_min_height_for_box(box) for box in self.group_boxes] + [0]
            max_height = max(heights)
            self.parent_dialog.setMinimumHeight(max_height)

    def clear_layout_down_to(self, level):
        def clear_box(box):
            layout = box.layout()
            for i in range(layout.count()):
                layout.itemAt(i).widget().deleteLater()

        import sip

        if level >= 0:
            special_flags_box = self.get_special_flags_from_box(self.boxes[level])
            clear_box(special_flags_box)
        for box in self.boxes[level + 1:]:
            sip.delete(box)
        self.boxes = self.boxes[:level + 1]
        self._using_dialog_box = False
        self.set_size()

    def find_box(self, box):
        return self.boxes.index(box)

    def setup_ui_at_level(self, level, level_state=None):
        if level_state:
            set_label, set_flag, set_special = level_state
        else:
            set_label = set_flag = set_special = None
        if level == -1:
            spec = self.spec
        else:
            # This is already getting new text selected.
            selected = self.get_texts_selected()[:level + 1]
            spec = self.spec[selected]
        if spec is None:
            return
        if type(spec) is list:
            special_flags_box = self.get_special_flags_from_box(self.boxes[level])
            self.fill_flags_box(special_flags_box, spec, state=set_special)
            return
        if type(spec) is str:
            if self.translate_flag(spec) != 'canvas':
                return
            group_box = self.make_pose_group_box(self.parent_dialog, self.parent_canvas)
            self._using_dialog_box = True
        else:
            labels = self.get_keys_from_label_dict(spec)
            flags = spec.get(self._flags, [])
            group_box = self.assemble_group_box(
                self.parent_dialog, labels, flags, set_label=set_label, set_flag=set_flag
            )
        self.splitter.addWidget(group_box)
        # Works even when level == -1, when self.boxes is empty.
        self.boxes = self.boxes[:level + 1] + [group_box]
        self.set_size()

    def init_ui(self):
        layout = QtWidgets.QVBoxLayout(self.parent_dialog)
        layout.addWidget(self.splitter)
        # buttons
        bb = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
            QtCore.Qt.Horizontal,
            self.parent_dialog
        )
        bb.button(bb.Ok).setIcon(labelme.utils.newIcon('done'))
        bb.button(bb.Cancel).setIcon(labelme.utils.newIcon('undo'))
        bb.accepted.connect(self.parent_dialog.validate)
        bb.rejected.connect(self.parent_dialog.reject)
        layout.addWidget(bb)

    def set_to_state(self, state=None, top_label=None):
        # Clear all layout
        self.clear_layout_down_to(-1)
        self.setting_state = True
        if state:
            for level, level_state in enumerate(state):
                self.setup_ui_at_level(level - 1, level_state=level_state)
        else:
            # Setup leftmost groupbox, select top_label in this groupbox,
            # and then setup next level groupbox.
            # top_label can be None, which is okay.
            self.setup_ui_at_level(-1, level_state=(top_label, None, None))
            if top_label:
                self.setup_ui_at_level(0)
        self.setting_state = False

    def get_first_edit(self):
        assert self.boxes
        return self.get_edit_from_box(self.boxes[0])

    def focus_on_first_edit(self):
        edit0 = self.get_first_edit()
        edit0.setFocus(QtCore.Qt.PopupFocusReason)


class LabelDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        from labelme.utils import Config

        super(LabelDialog, self).__init__(parent)

        self.ui = UILabelDialog(Config.get('labels'), self, parent)
        self.ui.init_ui()
        self._form = None
        self._shapes = None
        self._selected_shape = None
        self._boxes = []
        self._bindings = {}
        self._last_label = None

    @property
    def last_label(self):
        return self._last_label or ''

    @staticmethod
    def get_min_width_for_box(box):
        label_list = LabelDialog.get_list_from_box(box)
        return label_list.sizeHintForColumn(0) * 1.5

    @staticmethod
    def get_min_height_for_box(box):
        label_list = LabelDialog.get_list_from_box(box)
        return label_list.sizeHintForRow(0) * label_list.count() * 1.2

    def set_label_list_selected(self, label_list, text):
        items = label_list.findItems(text, QtCore.Qt.MatchFixedString)
        edit = self.get_edit_from_box(label_list.parent())
        if items:
            if len(items) != 1:
                logger.warning("Label list has duplicate '{}'".format(text))
            label_list.setCurrentItem(items[0])
            row = label_list.row(items[0])
            edit.completer().setCurrentRow(row)

    def label_selected(self, item):
        if self.ui.setting_state:
            return
        label_list = item.listWidget()
        box = label_list.parent()
        # Set text edit content to selected label
        edit = self.ui.get_edit_from_box(box)
        edit.setText(item.text())
        # Clear existing layout down to the level we're selecting items
        # and set up new ui
        level = self.ui.find_box(box)
        self.ui.clear_layout_down_to(level)
        self.ui.setup_ui_at_level(level)

    def validate(self):
        self._form, self._shapes = self.ui.get_full_state()
        if self._form is not None:
            self.accept()
        else:
            QtWidgets.QMessageBox.warning(self, 'Error', 'Label form is not complete')

    def popUp(self, selected_shape, text=None, move=True):
        self._form = self._shapes = None
        self._selected_shape = selected_shape
        self.ui.set_to_state(state=selected_shape.form, top_label=text)
        self.ui.focus_on_first_edit()
        if move:
            self.move(QtGui.QCursor.pos())

        self.exec_()
        if self._form:
            text = self.ui.get_first_label_from_state(self._form)
            self._last_label = text
            return text, self._form, self._shapes
        else:
            return None, None, None
