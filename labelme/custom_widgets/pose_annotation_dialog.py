from qtpy import QtCore, QtWidgets

import labelme.utils
from labelme.custom_widgets.preview_canvas import PreviewCanvas
from labelme.utils import Config


class UIPoseAnnotationDialog(object):
    def setupUi(self, dialog):
        dialog.setObjectName("Dialog")
        dialog.resize(800, 600)
        dialog.setModal(True)
        self.gridLayout = QtWidgets.QGridLayout(dialog)
        self.gridLayout.setObjectName("gridLayout")
        self.canvas = PreviewCanvas.from_canvas(dialog.parent(), no_highlight=True)
        self.canvas.setObjectName("canvas")
        self.point_label_list = QtWidgets.QListWidget(dialog)
        self.point_label_list.setObjectName("point_labels_list")
        self.buttonBox = QtWidgets.QDialogButtonBox(dialog)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel | QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.buttonBox.button(self.buttonBox.Ok).setIcon(labelme.utils.newIcon('done'))
        self.buttonBox.button(self.buttonBox.Cancel).setIcon(labelme.utils.newIcon('undo'))
        self.gridLayout.addWidget(self.point_label_list, 0, 1, 1, 1)
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
        # Results
        self.point_idx_to_label = selected_shape.get_annotation_result()
        self.label_to_point_idx = {v: k for k, v in self.point_idx_to_label.items()}
        raw_labels = Config.get('point_labels', [])
        self._point_labels = [self.get_label_list_text(i, l) for i, l in enumerate(raw_labels)]
        # Label list
        self.label_list = label_list = self.ui.point_label_list
        label_list.addItems(self._point_labels)
        label_list.setCurrentRow(0)
        # Canvas; only display selected shape
        self.canvas = canvas = self.ui.canvas
        # Returns a copy of the shape (which we'll modify later)
        self.selected_shape = canvas.replace_and_focus_shape(selected_shape)
        canvas.vertexSelected.connect(self.on_vertex_selected)

    def get_label_list_text(self, label_idx, label_str):
        base_str = f'{str(label_idx)}: {label_str}'
        if label_idx in self.label_to_point_idx:
            vertex_idx = self.label_to_point_idx[label_idx]
            base_str += f' -> Point {vertex_idx}'
        return base_str

    def on_vertex_selected(self) -> None:
        """
        On vertex selected, display label by vertex, display point in label list,
        and store this information.
        """
        total_row = self.label_list.count()
        current_row, current_item = self.label_list.currentRow(), self.label_list.currentItem()
        vertex_id = self.canvas.hVertex
        # Set point text in label list
        current_item.setText(self._point_labels[current_row] + f' -> Point {vertex_id}')
        if current_row + 1 < total_row:
            self.label_list.setCurrentRow(current_row + 1)
        # Set label to the shape on preview canvas (not the main canvas)
        # as we may need to undo this operation (like if dialog was reject()ed).
        self.selected_shape.set_vertex_label(vertex_id, current_row)
        # Store point -> label mapping
        self.point_idx_to_label[vertex_id] = current_row
