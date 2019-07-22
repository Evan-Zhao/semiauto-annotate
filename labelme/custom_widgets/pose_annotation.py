from labelme.shape import PoseShape
from qtpy import QtCore, QtWidgets

from labelme.utils import Config


class UIPoseAnnotationWidget(object):
    def __init__(self, canvas):
        self.gridLayout = None
        self.canvas = None
        self.point_label_list = None
        self.parent_canvas = canvas

    def setupUi(self, widget):
        self.gridLayout = QtWidgets.QGridLayout(widget)
        self.gridLayout.setObjectName("gridLayout")
        self.canvas = self.parent_canvas.to_preview_canvas()
        self.canvas.setObjectName("canvas")
        self.point_label_list = QtWidgets.QListWidget(widget)
        self.point_label_list.setObjectName("point_labels_list")
        self.gridLayout.addWidget(self.point_label_list, 0, 1, 1, 1)
        self.gridLayout.addWidget(self.canvas, 0, 2, 1, 1)
        QtCore.QMetaObject.connectSlotsByName(widget)


class PoseAnnotationWidget(QtWidgets.QWidget):
    def __init__(self, parent, canvas, selected_shape):
        super(PoseAnnotationWidget, self).__init__(parent)
        self.ui = UIPoseAnnotationWidget(canvas)
        self.ui.setupUi(self)
        raw_labels = Config.get('point_labels', [])
        self._point_labels = [self.get_label_list_text(i, l) for i, l in enumerate(raw_labels)]
        # Label list
        self.label_list = label_list = self.ui.point_label_list
        label_list.addItems(self._point_labels)
        label_list.setCurrentRow(0)
        # Canvas; only display points
        self.canvas = canvas = self.ui.canvas
        canvas.vertexSelected.connect(self.on_vertex_selected)
        for s in canvas.shapes:
            if s.shape_type != 'point' and s != selected_shape:
                canvas.setShapeVisible(s, False)
        # Shapes involved
        self.point_list = [None] * PoseShape.n_pose_points
        canvas.shapes.append(PoseShape(self.point_list))

    @staticmethod
    def get_label_list_text(label_idx, label_str):
        base_str = f'{str(label_idx)}: {label_str}'
        return base_str

    def get_shapes(self):
        return self.canvas.shapes

    def on_vertex_selected(self) -> None:
        """
        On vertex selected, display label by vertex, display point in label list,
        and store this information.
        """
        total_row = self.label_list.count()
        current_row, current_item = self.label_list.currentRow(), self.label_list.currentItem()
        if current_row + 1 < total_row:
            self.label_list.setCurrentRow(current_row + 1)

        shape = self.canvas._hShape
        shape_id = self.canvas.shapes.index(shape)
        self.canvas.shapes.pop(shape_id)
        self.point_list[current_row] = shape.points[0]
        self.canvas.shapes[-1] = PoseShape(self.point_list)
