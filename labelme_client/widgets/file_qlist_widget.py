import os.path as osp

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import QListWidget, QListWidgetItem


class FileQListWidget(QListWidget):
    load_file_signal = Signal(str)

    def __init__(self, uri_signal, get_uri, *args, **kwargs):
        super().__init__(*args, **kwargs)
        uri_signal.connect(self.handle_update)
        self.itemSelectionChanged.connect(self.file_selection_changed)
        self.currentItemChanged.connect(self.file_selection_changed)
        self.get_uri = get_uri

    def is_empty(self):
        return self.count() == 0

    @property
    def uri(self):
        return self.get_uri()

    def make_items(self, uri_file_list):
        self.clear()
        for filename, is_label in uri_file_list:
            item = QListWidgetItem(filename)
            item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            item.setCheckState(Qt.Checked if is_label else Qt.Unchecked)
            self.addItem(item)

    def handle_update(self):
        uri = self.uri
        if not uri:
            return
        self.make_items(self.uri.file_list)
        if uri.file_list:
            self.setCurrentRow(0)
        else:
            # This does not trigger itemSelectionChanged
            self.clearSelection()
            self.file_selection_changed()
        filename = uri[0]
        if filename:
            self.load_file_signal.emit(filename)

    def file_selection_changed(self):
        r = self.currentRow()
        if r == -1:
            # `None` will be translated to '',
            # so sending '' anyway
            self.load_file_signal.emit('')
            return
        filename = self.uri[r]
        if filename:
            self.load_file_signal.emit(filename)

    def accept_selection_change(self):
        r = self.currentRow()
        assert r is not None
        self.uri.set_selection(r)

    def revert_selection_change(self):
        self.itemSelectionChanged.disconnect()
        self.setCurrentRow(self.uri.idx_of_list)
        self.itemSelectionChanged.connect(self.file_selection_changed)

    def select_next(self):
        r = self.uri.get_next_idx()
        self.setCurrentRow(r)

    def select_prev(self):
        r = self.uri.get_prev_idx()
        self.setCurrentRow(r)

    def file_search_changed(self):
        self.uri.filter(self.sender().text())
        self.make_items(self.uri.file_list)

    def set_current_file_saved(self, saved_to_abs):
        self.uri.set_current_file_saved(saved_to_abs)
        self.make_items(self.uri.file_list)
