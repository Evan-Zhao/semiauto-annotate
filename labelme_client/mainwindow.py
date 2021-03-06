import functools
import os
import os.path as osp
import re
import webbrowser
from types import SimpleNamespace

from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets
from qtpy.QtCore import Qt, Signal

from labelme_base.shape import Shape
from . import __appname__, utils
from .config import Config
from .logger import logger
from .uri import URI
from .utils import LabelFile, ImageFile
from .widgets import Canvas, ColorDialog, EscapableQListWidget, \
    LabelDialog, LabelQListWidget, ToolBar, ZoomWidget, FileQListWidget


# FIXME
# - [medium] Set max zoom value to something big enough for FitWidth/Window

# TODO(unknown):
# - [high] Add polygon movement with arrow keys
# - [high] Deselect shape when clicking and already selected(?)
# - [low,maybe] Open images with drag & drop.
# - [low,maybe] Preview images on file dialogs.
# - Zoom is too "steppy".


class UIMainWindow(object):
    def __init__(self):
        self.labelList = None
        self.shape_dock = None
        self.uniqLabelList = None
        self.label_dock = None
        self.fileSearch = None
        self.fileListWidget = None
        self.file_dock = None
        self.zoomWidget = None
        self.colorDialog = None
        self.canvas = None
        self.labelDialog = None
        self.scrollBars = None
        self.menus = None
        self.tools = None

    def setup_ui(self, window):
        self.labelList = LabelQListWidget(window)
        self.labelList.itemActivated.connect(window.labelSelectionChanged)
        self.labelList.itemSelectionChanged.connect(window.labelSelectionChanged)
        self.labelList.itemDoubleClicked.connect(window.editLabel)
        # Connect to itemChanged to detect checkbox changes.
        self.labelList.itemChanged.connect(window.labelItemChanged)
        self.labelList.setDragDropMode(
            QtWidgets.QAbstractItemView.InternalMove)

        self.shape_dock = QtWidgets.QDockWidget('Polygon Labels', window)
        self.shape_dock.setObjectName('Labels')
        self.shape_dock.setWidget(self.labelList)

        self.uniqLabelList = EscapableQListWidget()
        self.uniqLabelList.setToolTip(
            "Select label to start annotating for it. "
            "Press 'Esc' to deselect.")
        if Config.get('labels'):
            self.uniqLabelList.addItems(set(Config.get('labels')) - {'__flags'})
            self.uniqLabelList.sortItems()
        self.label_dock = QtWidgets.QDockWidget(u'Label List', window)
        self.label_dock.setObjectName(u'Label List')
        self.label_dock.setWidget(self.uniqLabelList)

        self.fileSearch = QtWidgets.QLineEdit()
        self.fileSearch.setPlaceholderText('Search Filename')
        self.fileListWidget = FileQListWidget(
            window.bound_to_new_uri,
            lambda: window.uri
        )
        self.fileListWidget.load_file_signal.connect(window.on_file_changed)
        self.fileSearch.textChanged.connect(self.fileListWidget.file_search_changed)
        fileListLayout = QtWidgets.QVBoxLayout()
        fileListLayout.setContentsMargins(0, 0, 0, 0)
        fileListLayout.setSpacing(0)
        fileListLayout.addWidget(self.fileSearch)
        fileListLayout.addWidget(self.fileListWidget)
        self.file_dock = QtWidgets.QDockWidget(u'File List', window)
        self.file_dock.setObjectName(u'Files')
        fileListSearchWidget = QtWidgets.QWidget()
        fileListSearchWidget.setLayout(fileListLayout)
        self.file_dock.setWidget(fileListSearchWidget)

        self.zoomWidget = ZoomWidget()
        self.colorDialog = ColorDialog(parent=window)

        self.canvas = self.labelList.canvas = Canvas()
        self.canvas.zoomRequest.connect(window.zoomRequest)

        self.labelDialog = LabelDialog(parent=self.canvas)

        scrollArea = QtWidgets.QScrollArea()
        scrollArea.setWidget(self.canvas)
        scrollArea.setWidgetResizable(True)
        self.scrollBars = {
            Qt.Vertical: scrollArea.verticalScrollBar(),
            Qt.Horizontal: scrollArea.horizontalScrollBar(),
        }
        self.canvas.scrollRequest.connect(window.scrollRequest)
        self.canvas.newShape.connect(window.newShape)
        self.canvas.shapeMoved.connect(window.setDirty)
        self.canvas.selectionChanged.connect(window.shapeSelectionChanged)
        self.canvas.drawingPolygon.connect(window.action_storage.refresh_all)
        self.canvas.mergeShape.connect(window.mergeShapes)

        window.setCentralWidget(scrollArea)

        features = QtWidgets.QDockWidget.DockWidgetFeatures()
        for dock in ['label_dock', 'shape_dock', 'file_dock']:
            if Config.get((dock, 'closable')):
                features = features | QtWidgets.QDockWidget.DockWidgetClosable
            if Config.get((dock, 'floatable')):
                features = features | QtWidgets.QDockWidget.DockWidgetFloatable
            if Config.get((dock, 'movable')):
                features = features | QtWidgets.QDockWidget.DockWidgetMovable
            getattr(self, dock).setFeatures(features)
            if Config.get((dock, 'show')) is False:
                getattr(self, dock).setVisible(False)

        window.addDockWidget(Qt.RightDockWidgetArea, self.label_dock)
        window.addDockWidget(Qt.RightDockWidgetArea, self.shape_dock)
        window.addDockWidget(Qt.RightDockWidgetArea, self.file_dock)

        shortcuts = Config.get('shortcuts')
        self.zoomWidget.setWhatsThis(
            'Zoom in or out of the image. Also accessible with '
            '{} and {} from the canvas.'
                .format(
                utils.fmtShortcut(
                    '{},{}'.format(
                        shortcuts['zoom_in'], shortcuts['zoom_out']
                    )
                ),
                utils.fmtShortcut("Ctrl+Wheel"),
            )
        )
        self.zoomWidget.setEnabled(False)
        self.zoomWidget.valueChanged.connect(window.paintCanvas)

        # Lavel list context menu.
        labelMenu = QtWidgets.QMenu()
        self.labelList.setContextMenuPolicy(Qt.CustomContextMenu)
        self.labelList.customContextMenuRequested.connect(window.popLabelListMenu)
        self.menus = SimpleNamespace(
            file=self.menu(window, '&File'),
            edit=self.menu(window, '&Edit'),
            view=self.menu(window, '&View'),
            help=self.menu(window, '&Help'),
            recentFiles=QtWidgets.QMenu('Open &Recent'),
            labelList=labelMenu,
        )
        self.menus.file.aboutToShow.connect(window.updateFileMenu)
        self.tools = self.toolbar(window, 'Tools')

    @staticmethod
    def menu(window, title, actions=None):
        menu = window.menuBar().addMenu(title)
        if actions:
            utils.addActions(menu, actions)
        return menu

    @staticmethod
    def toolbar(window, title, actions=None):
        toolbar = ToolBar(title)
        toolbar.setObjectName('%sToolBar' % title)
        # toolbar.setOrientation(Qt.Vertical)
        toolbar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        if actions:
            utils.addActions(toolbar, actions)
        window.addToolBar(Qt.LeftToolBarArea, toolbar)
        return toolbar


class MainWindow(QtWidgets.QMainWindow):
    FIT_WINDOW, FIT_WIDTH, MANUAL_ZOOM = 0, 1, 2
    bound_to_new_uri = Signal(URI)
    max_recent = 7

    def __init__(
            self,
            file_or_folder,
            output_file=None,
            output_dir=None,
    ):
        super(MainWindow, self).__init__()

        self.action_storage = utils.ActionStorage(self)
        # Whether we need to save or not.
        self.dirty = False
        self.label_file = None
        if output_file:
            if Config.get('auto_save'):
                logger.warn(
                    'If `auto_save` argument is True, `output_file` argument '
                    'is ignored and output filename is automatically '
                    'set as IMAGE_BASENAME.json.'
                )
            self.label_file = output_file
        # Ignore signal flags
        self._noSelectionSlot = False
        # Last open directory
        self.lastOpenDir = None
        self._uri = None
        self.output_dir = output_dir
        self.zoomMode = self.FIT_WINDOW
        self.scalers = {
            self.FIT_WINDOW: self.scaleFitWindow,
            self.FIT_WIDTH: self.scaleFitWidth,
            # Set to one to scale to 100% when loading files.
            self.MANUAL_ZOOM: lambda: 1,
        }

        # Application state.
        # Restore application settings.
        # FIXME: QSettings.value can return None on PyQt4
        self.settings = QtCore.QSettings('labelme_client', 'labelme_client')
        self.recentFiles = self.settings.value('recentFiles', []) or []
        self.lineColor = self.settings.value('line/color', None)
        self.fillColor = self.settings.value('fill/color', None)
        geometry = self.settings.value('window/geometry')
        if geometry:
            self.restoreGeometry(geometry)
        self.restoreState(self.settings.value('window/state', QtCore.QByteArray()))

        self.ui = UIMainWindow()
        self.ui.setup_ui(self)

        self.statusBar().showMessage('%s started.' % __appname__)
        self.statusBar().show()
        if Config.get('file_search'):
            self.fileSearch.setText(Config.get('file_search'))
        # Populate actions in menus
        self.populate_actions()
        # Populate the File menu dynamically.
        self.updateFileMenu()
        # Set and load URI
        self.uri = URI.from_file_or_folder(file_or_folder, output_dir) \
            if file_or_folder else None

    def __getattr__(self, item):
        return getattr(self.ui, item)

    @property
    def filename(self):
        return self.uri.filename if self.uri else None

    @property
    def uri(self):
        return self._uri

    @uri.setter
    def uri(self, value):
        self._uri = value
        if value:
            self.bound_to_new_uri.emit(value)

    @property
    def title(self):
        title = __appname__
        if self.filename is not None:
            dirty = '*' if self.dirty else ''
            title = f'{title} - {self.filename}{dirty}'
        return title

    def get_or_make_label_file(self):
        if self.label_file is None:
            self.label_file = LabelFile()
        return self.label_file

    # Support Functions

    def setDirty(self):
        if Config.get('auto_save') or self.saveAuto.isChecked():
            label_file = osp.splitext(self.filename)[0] + '.json'
            if self.output_dir:
                label_file_without_path = osp.basename(label_file)
                label_file = osp.join(self.output_dir, label_file_without_path)
            self.saveLabels(label_file)
            return
        self.dirty = True
        self.action_storage.refresh_all()
        self.setWindowTitle(self.title)

    def setClean(self):
        self.dirty = False
        self.action_storage.refresh_all()
        self.setWindowTitle(self.title)

    def queueEvent(self, function):
        QtCore.QTimer.singleShot(0, function)

    def status(self, message, delay=5000):
        self.statusBar().showMessage(message, delay)

    def resetState(self):
        self.labelList.clear()
        self.label_file = None
        self.canvas.clear_all()

    def addRecentFile(self, filename):
        if filename in self.recentFiles:
            self.recentFiles.remove(filename)
        elif len(self.recentFiles) >= self.max_recent:
            self.recentFiles.pop()
        self.recentFiles.insert(0, filename)

    # Callbacks

    def undoShapeEdit(self):
        if self.canvas.restoreShape():
            self.setDirty()
            self.labelList.clear()
            self.loadShapes(self.canvas.shapes)
        self.action_storage.refresh_all()

    def tutorial(self):
        url = 'https://github.com/wkentaro/labelme/tree/master/examples/tutorial'  # NOQA
        webbrowser.open(url)

    def toggleDrawMode(self, edit=True, createMode='polygon'):
        self.canvas.setEditing(edit)
        self.canvas.createMode = createMode
        self.action_storage.refresh_all()

    def setEditMode(self):
        self.toggleDrawMode(True)

    def updateFileMenu(self):
        current = self.filename

        def exists(filename):
            return osp.exists(str(filename))

        menu = self.menus.recentFiles
        menu.clear()
        files = [f for f in self.recentFiles if f != current and exists(f)]
        for i, f in enumerate(files):
            icon = utils.newIcon('labels')
            action = QtWidgets.QAction(
                icon, '&%d %s' % (i + 1, QtCore.QFileInfo(f).fileName()), self)
            action.triggered.connect(functools.partial(self.loadRecent, f))
            menu.addAction(action)

    def popLabelListMenu(self, point):
        self.menus.labelList.exec_(self.labelList.mapToGlobal(point))

    def validateLabel(self, label):
        # no validation
        if Config.get('validate_label') is None:
            return True

        for i in range(self.uniqLabelList.count()):
            label_i = self.uniqLabelList.item(i).text()
            if Config.get('validate_label') in ['exact', 'instance']:
                if label_i == label:
                    return True
            if Config.get('validate_label') == 'instance':
                m = re.match(r'^{}-[0-9]*$'.format(label_i), label)
                if m:
                    return True
        return False

    def editLabel(self, item=False):
        if item and not isinstance(item, QtWidgets.QListWidgetItem):
            raise TypeError('unsupported type of item: {}'.format(type(item)))

        if not self.canvas.editing():
            return
        if not item:
            item = self.labelList.currentItem()
        if item is None:
            return
        shape = self.labelList.get_shape_from_item(item)
        if shape is None:
            return
        is_new_label, text, form, shapes = self.labelDialog.popUp(shape)
        if text is None:
            return
        if not self.validateLabel(text):
            self.errorMessage('Invalid label',
                              "Invalid label '{}' with validation type '{}'"
                              .format(text, Config.get('validate_label')))
            return
        if shapes:
            self.canvas.loadShapes(shapes)
            self.loadShapes(shapes)
        else:
            shape.set_label(form)
        item.setText(text)
        self.setDirty()
        if is_new_label:
            self.uniqLabelList.addItem(text)
            self.uniqLabelList.sortItems()

    # React to canvas signals.

    def shapeSelectionChanged(self):
        self._noSelectionSlot = True
        self.labelList.clearSelection()
        for shape in self.canvas.selectedShapes:
            item = self.labelList.get_item_from_shape(shape)
            item.setSelected(True)
        self._noSelectionSlot = False
        self.action_storage.refresh_all()

    def mergeShapes(self, all_removed, added):
        self.remLabels(all_removed)
        self.addLabel(added)
        self.setDirty()

    def addLabel(self, shape, is_new_label=False):
        item = QtWidgets.QListWidgetItem(shape.label)
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(Qt.Checked)
        self.labelList.itemsToShapes.append((item, shape))
        self.labelList.addItem(item)
        if is_new_label:
            self.uniqLabelList.addItem(shape.label)
            self.uniqLabelList.sortItems()

    def remLabels(self, shapes):
        for shape in shapes:
            item = self.labelList.get_item_from_shape(shape)
            self.labelList.takeItem(self.labelList.row(item))

    def loadShapes(self, shapes):
        self._noSelectionSlot = True
        for shape in shapes:
            self.addLabel(shape)
        self.labelList.clearSelection()
        self._noSelectionSlot = False

    def saveLabels(self, abs_path):
        try:
            if osp.dirname(abs_path) and not osp.exists(osp.dirname(abs_path)):
                os.makedirs(osp.dirname(abs_path))
            self.get_or_make_label_file().save(self.snapshot, filename=abs_path)
            self.fileListWidget.set_current_file_saved(abs_path)
            return True
        except utils.LabelFileError as e:
            self.errorMessage('Error saving label data', '<b>%s</b>' % e)
            return False

    def copySelectedShape(self):
        from qtpy.QtGui import QGuiApplication

        shapes_json = self.canvas.copySelectedShapes()
        clipboard = QGuiApplication.clipboard()
        clipboard.setText(shapes_json)

    def pasteShape(self):
        from qtpy.QtGui import QGuiApplication

        clipboard = QGuiApplication.clipboard()
        if not clipboard.ownsClipboard():
            return
        shapes_json = clipboard.text()
        added_shapes = self.canvas.pasteShapes(shapes_json)
        self.labelList.clearSelection()
        for shape in added_shapes:
            self.addLabel(shape)
        self.setDirty()

    def labelSelectionChanged(self):
        if self._noSelectionSlot:
            return
        if self.canvas.editing():
            selected_shapes = []
            for item in self.labelList.selectedItems():
                shape = self.labelList.get_shape_from_item(item)
                selected_shapes.append(shape)
            if selected_shapes:
                self.canvas.selectShapes(selected_shapes)

    def labelItemChanged(self, item):
        shape = self.labelList.get_shape_from_item(item)
        label = str(item.text())
        if label != shape.label:
            shape.label = str(item.text())
            self.setDirty()
        else:  # User probably changed item visibility
            self.canvas.setShapeVisible(shape, item.checkState() == Qt.Checked)

    # Callback functions:

    def newShape(self):
        """Pop-up and give focus to the label editor.

        position MUST be in global coordinates.
        """
        items = self.uniqLabelList.selectedItems()
        text = None
        if items:
            text = items[0].text()
        if Config.get('display_label_popup') or not text:
            # instance label auto increment
            if Config.get('instance_label_auto_increment'):
                previous_label = self.labelDialog.last_label
                split = previous_label.split('-')
                if len(split) > 1 and split[-1].isdigit():
                    split[-1] = str(int(split[-1]) + 1)
                    instance_text = '-'.join(split)
                else:
                    instance_text = previous_label
                if instance_text != '':
                    text = instance_text
            is_new_label, text, form, shapes = \
                self.labelDialog.popUp(self.canvas.shapes[-1], text=text)

        if text and not self.validateLabel(text):
            self.errorMessage('Invalid label',
                              "Invalid label '{}' with validation type '{}'"
                              .format(text, Config.get('validate_label')))
            text = ''
        if text:
            self.labelList.clearSelection()
            self.addLabel(self.canvas.setLastLabel(form, shapes), is_new_label)
            self.action_storage.refresh_all()
            self.setDirty()
        else:
            self.canvas.undoLastLine()
            self.canvas.shapesBackups.pop()

    def scrollRequest(self, delta, orientation):
        units = - delta * 0.1  # natural scroll
        bar = self.scrollBars[orientation]
        bar.setValue(bar.value() + bar.singleStep() * units)

    def setZoom(self, value):
        self.zoomMode = self.MANUAL_ZOOM
        self.zoomWidget.setValue(value)

    def addZoom(self, increment=1.1):
        self.setZoom(self.zoomWidget.value() * increment)

    def zoomRequest(self, delta, pos):
        canvas_width_old = self.canvas.width()
        units = 1.1
        if delta < 0:
            units = 0.9
        self.addZoom(units)

        canvas_width_new = self.canvas.width()
        if canvas_width_old != canvas_width_new:
            canvas_scale_factor = canvas_width_new / canvas_width_old

            x_shift = round(pos.x() * canvas_scale_factor) - pos.x()
            y_shift = round(pos.y() * canvas_scale_factor) - pos.y()

            self.scrollBars[Qt.Horizontal].setValue(
                self.scrollBars[Qt.Horizontal].value() + x_shift)
            self.scrollBars[Qt.Vertical].setValue(
                self.scrollBars[Qt.Vertical].value() + y_shift)

    def setFitWindow(self, value=True):
        self.zoomMode = self.FIT_WINDOW if value else self.MANUAL_ZOOM
        self.adjustScale()

    def setFitWidth(self, value=True):
        self.zoomMode = self.FIT_WIDTH if value else self.MANUAL_ZOOM
        self.adjustScale()

    def togglePolygons(self, value):
        for item, shape in self.labelList.itemsToShapes:
            item.setCheckState(Qt.Checked if value else Qt.Unchecked)

    def on_file_changed(self, filename):
        self.resetState()
        if filename != '':
            if self.load_file(filename):
                self.sender().accept_selection_change()
            else:
                self.sender().revert_selection_change()
        self.setWindowTitle(self.title)

    def load_file(self, filename):
        """Load the specified file, or the last opened file if None."""

        def print_file_error(exception, filepath):
            self.errorMessage(
                'Error opening file',
                "<p><b>%s</b></p>"
                "<p>Make sure <i>%s</i> is a valid label file."
                % (exception, filepath))
            self.status("Error reading %s" % filepath)

        def print_image_unsupported_error(filename):
            formats = ['*.{}'.format(fmt.data().decode())
                       for fmt in QtGui.QImageReader.supportedImageFormats()]
            self.errorMessage(
                'Error opening file',
                '<p>Make sure <i>{0}</i> is a valid image file.<br/>'
                'Supported image formats: {1}</p>'
                    .format(filename, ','.join(formats)))
            self.status("Error reading %s" % filename)

        from .utils import ImageFileIOError, LabelFileError, ImageUnsupportedError

        self.canvas.setEnabled(False)
        if not osp.exists(filename):
            self.errorMessage(
                'Error opening file', 'No such file: <b>%s</b>' % filename)
            return False
        self.status("Loading %s..." % osp.basename(filename))
        if LabelFile.is_label_file(filename):
            try:
                label_file = LabelFile(filename)
            except LabelFileError as e:
                print_file_error(e, filename)
                return False
            self.label_file = label_file
            self.snapshot = label_file.main_snapshot
        else:
            try:
                image_file = ImageFile(filename)
            except ImageFileIOError as e:
                print_file_error(e, filename)
                return False
            except ImageUnsupportedError:
                print_image_unsupported_error(filename)
                return False
            self.canvas.load_image_file(image_file)
            self.loadShapes(self.canvas.shapes)
        self.setClean()
        self.canvas.setEnabled(True)
        self.adjustScale(initial=True)
        self.paintCanvas()
        self.action_storage.refresh_all()
        self.status("Loaded %s" % osp.basename(str(filename)))
        return True

    def resizeEvent(self, event):
        if self.canvas and not self.canvas.is_empty() \
                and self.zoomMode != self.MANUAL_ZOOM:
            self.adjustScale()
        super(MainWindow, self).resizeEvent(event)

    def paintCanvas(self):
        self.canvas.scale = 0.01 * self.zoomWidget.value()
        self.canvas.adjustSize()
        self.canvas.update()

    def adjustScale(self, initial=False):
        value = self.scalers[self.FIT_WINDOW if initial else self.zoomMode]()
        self.zoomWidget.setValue(int(100 * value))

    def scaleFitWindow(self):
        """Figure out the size of the pixmap to fit the main widget."""
        e = 2.0  # So that no scrollbars are generated.
        w1 = self.centralWidget().width() - e
        h1 = self.centralWidget().height() - e
        a1 = w1 / h1
        # Calculate a new scale value based on the pixmap's aspect ratio.
        w2 = self.canvas.pixmap.width() - 0.0
        h2 = self.canvas.pixmap.height() - 0.0
        a2 = w2 / h2
        return w1 / w2 if a2 >= a1 else h1 / h2

    def scaleFitWidth(self):
        # The epsilon does not seem to work too well here.
        w = self.centralWidget().width() - 2.0
        return w / self.canvas.pixmap.width()

    def closeEvent(self, event):
        if not self.mayContinue():
            event.ignore()
        self.settings.setValue(
            'filename', self.filename if self.filename else '')
        self.settings.setValue('window/geometry', self.saveGeometry())
        self.settings.setValue('window/state', self.saveState())
        self.settings.setValue('recentFiles', self.recentFiles)
        # ask the use for where to save the labels

    # User Dialogs #

    def loadRecent(self, filename):
        if self.mayContinue():
            self.uri = URI.from_file(filename)
            self.addRecentFile(filename)

    def openPrevImg(self, _value=False):
        keep_prev = Config.get('keep_prev')
        if QtGui.QGuiApplication.keyboardModifiers() == \
                (QtCore.Qt.ControlModifier | QtCore.Qt.ShiftModifier):
            Config.set('keep_prev', True)

        if not self.mayContinue():
            return
        self.ui.fileListWidget.select_prev()
        Config.set('keep_prev', keep_prev)

    def openNextImg(self, _value=False):
        keep_prev = Config.get('keep_prev')
        if QtGui.QGuiApplication.keyboardModifiers() == \
                (QtCore.Qt.ControlModifier | QtCore.Qt.ShiftModifier):
            Config.set('keep_prev', True)

        if not self.mayContinue():
            return
        self.ui.fileListWidget.select_next()
        Config.set('keep_prev', keep_prev)

    def openFile(self, _value=False):
        if not self.mayContinue():
            return
        path = self.uri.folder_name if self.uri else '.'
        formats = ['*.{}'.format(fmt.data().decode())
                   for fmt in QtGui.QImageReader.supportedImageFormats()]
        filters = "Image & Label files (%s)" % ' '.join(
            formats + ['*%s' % LabelFile.suffix]
        )
        filename = QtWidgets.QFileDialog.getOpenFileName(
            self, '%s - Choose Image or Label file' % __appname__,
            path, filters
        )
        filename = str(filename[0])
        if filename:
            self.uri = URI.from_file(filename)

    def changeOutputDirDialog(self, _value=False):
        default_output_dir = self.output_dir or (
            self.uri.folder_name if self.uri else None
        ) or '.'

        output_dir = QtWidgets.QFileDialog.getExistingDirectory(
            self, '%s - Save/Load Annotations in Directory' % __appname__,
            default_output_dir,
                  QtWidgets.QFileDialog.ShowDirsOnly |
                  QtWidgets.QFileDialog.DontResolveSymlinks,
        )
        output_dir = str(output_dir)

        if not output_dir:
            return

        self.output_dir = output_dir
        self.fileListWidget.output_dir_changed(output_dir)

        self.statusBar().showMessage(
            '%s . Annotations will be saved/loaded in %s' %
            ('Change Annotations Dir', self.output_dir))
        self.statusBar().show()

    def saveFile(self, _value=False):
        if self.hasLabels():
            if self.label_file:
                # DL20180323 - overwrite when in directory
                self._saveFile(self.label_file.filename)
            else:
                self._saveFile(self.saveFileDialog())

    def saveFileAs(self, _value=False):
        if self.hasLabels():
            self._saveFile(self.saveFileDialog())

    def saveFileDialog(self):
        caption = '%s - Choose File' % __appname__
        filters = 'Label files (*%s)' % LabelFile.suffix
        if self.output_dir:
            dlg = QtWidgets.QFileDialog(
                self, caption, self.output_dir, filters
            )
        else:
            dlg = QtWidgets.QFileDialog(
                self, caption, self.currentPath(), filters
            )
        dlg.setDefaultSuffix(LabelFile.suffix[1:])
        dlg.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
        dlg.setOption(QtWidgets.QFileDialog.DontConfirmOverwrite, False)
        dlg.setOption(QtWidgets.QFileDialog.DontUseNativeDialog, False)
        basename = osp.basename(osp.splitext(self.filename)[0])
        if self.output_dir:
            default_labelfile_name = osp.join(
                self.output_dir, basename + LabelFile.suffix
            )
        else:
            default_labelfile_name = osp.join(
                self.currentPath(), basename + LabelFile.suffix
            )
        filename = dlg.getSaveFileName(
            self, 'Choose File', default_labelfile_name,
            'Label files (*%s)' % LabelFile.suffix)
        filename = str(filename[0])
        return filename

    def _saveFile(self, filename):
        if filename and self.saveLabels(filename):
            self.addRecentFile(filename)
            self.setClean()

    def closeFile(self, _value=False):
        if not self.mayContinue():
            return
        self.resetState()
        self.setClean()
        self.action_storage.refresh_all()
        self.canvas.setEnabled(False)

    def getLabelFile(self):
        if self.filename.lower().endswith('.json'):
            label_file = self.filename
        else:
            label_file = osp.splitext(self.filename)[0] + '.json'

        return label_file

    def deleteFile(self):
        mb = QtWidgets.QMessageBox
        msg = 'You are about to permanently delete this label file, ' \
              'proceed anyway?'
        answer = mb.warning(self, 'Attention', msg, mb.Yes | mb.No)
        if answer != mb.Yes:
            return

        label_file = self.getLabelFile()
        if osp.exists(label_file):
            os.remove(label_file)
            logger.info('Label file is removed: {}'.format(label_file))

            item = self.fileListWidget.currentItem()
            item.setCheckState(Qt.Unchecked)

            self.resetState()

    # Message Dialogs. #
    def hasLabels(self):
        if not self.labelList.itemsToShapes:
            self.errorMessage(
                'No objects labeled',
                'You must label at least one object to save the file.')
            return False
        return True

    def hasLabelFile(self):
        if self.filename is None:
            return False

        label_file = self.getLabelFile()
        return osp.exists(label_file)

    def mayContinue(self):
        if not self.dirty:
            return True
        mb = QtWidgets.QMessageBox
        msg = 'Save annotations to "{}" before closing?'.format(self.filename)
        answer = mb.question(self,
                             'Save annotations?',
                             msg,
                             mb.Save | mb.Discard | mb.Cancel,
                             mb.Save)
        if answer == mb.Discard:
            return True
        elif answer == mb.Save:
            self.saveFile()
            return True
        else:  # answer == mb.Cancel
            return False

    def errorMessage(self, title, message):
        return QtWidgets.QMessageBox.critical(
            self, title, '<p><b>%s</b></p>%s' % (title, message))

    def currentPath(self):
        return osp.dirname(str(self.filename)) if self.filename else '.'

    def chooseColor1(self):
        accepted, color = self.colorDialog.getColor(
            self.lineColor, 'Choose line color', default=None
        )
        if accepted:
            self.lineColor = color
            self.settings.setValue('line/color', self.lineColor)
            self.canvas.update()

    def chooseColor2(self):
        accepted, color = self.colorDialog.getColor(
            self.fillColor, 'Choose fill color', default=None
        )
        if accepted:
            self.fillColor = color
            self.settings.setValue('fill/color', self.fillColor)
            self.canvas.update()

    def toggleKeepPrevMode(self):
        Config.set('keep_prev', not Config.get('keep_prev'))

    def deleteSelectedShape(self):
        yes, no = QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No
        msg = 'You are about to permanently delete {} polygons, ' \
              'proceed anyway?'.format(len(self.canvas.selectedShapes))
        if yes == QtWidgets.QMessageBox.warning(self, 'Attention', msg,
                                                yes | no):
            self.remLabels(self.canvas.deleteSelected())
            self.setDirty()
            self.action_storage.refresh_all()

    def copyShape(self):
        self.canvas.endMove(copy=True)
        self.labelList.clearSelection()
        for shape in self.canvas.selectedShapes:
            self.addLabel(shape)
        self.setDirty()

    def moveShape(self):
        self.canvas.endMove(copy=False)
        self.setDirty()

    def openDirDialog(self):
        if not self.mayContinue():
            return

        default_folder = self.uri.folder_name if self.uri else '.'

        targetDirPath = str(QtWidgets.QFileDialog.getExistingDirectory(
            self, '%s - Open Directory' % __appname__, default_folder,
                  QtWidgets.QFileDialog.ShowDirsOnly |
                  QtWidgets.QFileDialog.DontResolveSymlinks))
        self.uri = URI.from_folder(targetDirPath, self.output_dir)

    @property
    def snapshot(self):
        return self.canvas.snapshot

    @snapshot.setter
    def snapshot(self, value):
        self.canvas.snapshot = value
        self.loadShapes(self.canvas.shapes)

    def populate_actions(self):
        def zoom_enabled():
            return not self.canvas.is_empty()

        def create_mode_enabled(self_mode):
            if self.canvas.is_empty():
                return False
            return self.canvas.createMode != self_mode or self.canvas.editing()

        # Actions
        action = self.action_storage.make_action
        make_group = self.action_storage.make_group
        shortcuts = Config.get('shortcuts')

        # File menu
        open_ = action('&Open', self.openFile, shortcuts['open'], 'open',
                       'Open image or label file')
        opendir = action('&Open Dir', self.openDirDialog,
                         shortcuts['open_dir'], 'open', u'Open Dir')
        openNextImg = action(
            '&Next Image', self.openNextImg, shortcuts['open_next'],
            'next', u'Open next (hold Ctl+Shift to copy labels)',
            enable_condition=lambda: not self.fileListWidget.is_empty()
        )
        openPrevImg = action(
            '&Prev Image', self.openPrevImg, shortcuts['open_prev'],
            'prev', u'Open prev (hold Ctl+Shift to copy labels)',
            enable_condition=lambda: not self.fileListWidget.is_empty()
        )
        save = action(
            '&Save', self.saveFile, shortcuts['save'], 'save',
            'Save labels to file',
            enable_condition=lambda: self.dirty
        )
        saveAs = action(
            '&Save As', self.saveFileAs, shortcuts['save_as'],
            'save-as', 'Save labels to a different file',
            enable_condition=lambda: not self.canvas.is_empty()
        )
        saveAuto = action(
            text='Save &Automatically',
            slot=lambda x: self.saveAuto.setChecked(x),
            icon='save', tip='Save automatically', checkable=True
        )
        self.saveAuto = saveAuto
        saveAuto.setChecked(Config.get('auto_save'))
        changeOutputDir = action(
            '&Change Output Dir',
            slot=self.changeOutputDirDialog,
            shortcut=shortcuts['save_to'],
            icon='open',
            tip=u'Change where annotations are loaded/saved'
        )
        close = action(
            '&Close', self.closeFile, shortcuts['close'],
            'close', 'Close current file',
            enable_condition=lambda: not self.canvas.is_empty()
        )
        deleteFile = action(
            '&Delete File', self.deleteFile, shortcuts['delete_file'],
            'delete', 'Delete current label file',
            enable_condition=lambda: self.hasLabelFile()
        )

        quit_ = action('&Quit', self.close, shortcuts['quit'], 'quit', 'Quit application')
        file_menu = (
            (
                open_, opendir, openNextImg, openPrevImg,
                self.menus.recentFiles, save, saveAs, saveAuto,
                changeOutputDir, close, deleteFile
            ),
            (quit_,)
        )

        # View menu
        fill_drawing = action(
            'Fill Drawing Polygon', lambda x: self.canvas.setFillDrawing(x),
            None, 'color', 'Fill polygon while drawing',
            checkable=True, enabled=True
        )

        hideAll = action(
            '&Hide\nPolygons', functools.partial(self.togglePolygons, False),
            icon='eye', tip='Hide all polygons',
            enable_condition=lambda: self.canvas.has_shapes()
        )
        showAll = action(
            '&Show\nPolygons', functools.partial(self.togglePolygons, True),
            icon='eye', tip='Show all polygons',
            enable_condition=lambda: self.canvas.has_shapes()
        )

        zoomIn = action(
            'Zoom &In', functools.partial(self.addZoom, 1.1), shortcuts['zoom_in'],
            'zoom-in', 'Increase zoom level', enable_condition=zoom_enabled)
        zoomOut = action(
            '&Zoom Out', functools.partial(self.addZoom, 0.9), shortcuts['zoom_out'],
            'zoom-out', 'Decrease zoom level', enable_condition=zoom_enabled
        )

        zoomOrg = action(
            '&Original size', functools.partial(self.setZoom, 100), shortcuts['zoom_to_original'],
            'zoom', 'Zoom to original size', checkable=True, enable_condition=zoom_enabled
        )
        fitWindow = action(
            '&Fit Window', self.setFitWindow, shortcuts['fit_window'],
            'fit-window', 'Zoom follows window size',
            checkable=True, enable_condition=zoom_enabled
        )
        fitWidth = action(
            'Fit &Width', self.setFitWidth, shortcuts['fit_width'],
            'fit-width', 'Zoom follows window width',
            checkable=True, enable_condition=zoom_enabled
        )
        fit_group = make_group(zoomOrg, fitWindow, fitWidth, exclusive=True)
        fitWindow.setChecked(Qt.Checked)

        view_menu = (
            (
                self.ui.label_dock.toggleViewAction(),
                self.ui.shape_dock.toggleViewAction(),
                self.ui.file_dock.toggleViewAction()
            ),
            (fill_drawing,),
            (hideAll, showAll),
            (zoomIn, zoomOut),
            fit_group
        )

        # Edit menu
        allCreateModes = tuple((
            action(
                f'Create {mode}',
                functools.partial(self.toggleDrawMode, False, createMode=mode),
                shortcuts[f'create_{mode}'], 'objects', f'Start drawing {mode}',
                enable_condition=functools.partial(create_mode_enabled, mode)
            )
            for mode in Shape.all_types
        ))

        edit = action(
            '&Edit Label', self.editLabel, shortcuts['edit_label'],
            'edit', 'Modify the label of the selected polygon',
            enable_condition=lambda: len(self.canvas.selectedShapes) == 1
        )
        copy = action(
            'Copy Polygons', self.copySelectedShape, shortcuts['copy'],
            'copy', 'Copy selected polygons to clipboard',
            enable_condition=lambda: bool(self.canvas.selectedShapes)
        )
        paste = action(
            'Paste Polygons', self.pasteShape, shortcuts['paste'],
            'paste', 'Paste polygon on canvas',
            enable_condition=lambda: not self.canvas.is_empty()
        )
        delete = action(
            'Delete Polygons', self.deleteSelectedShape, shortcuts['delete_polygon'],
            'cancel', 'Delete the selected polygons',
            enable_condition=lambda:
            bool(self.canvas.selectedShapes) and not self.canvas.drawing()
        )

        undo = action(
            'Undo', self.undoShapeEdit, shortcuts['undo'],
            'undo', 'Undo last add and edit of shape',
            enable_condition=lambda:
            self.canvas.isShapeRestorable and not self.canvas.drawing()
        )
        undoLastPoint = action(
            'Undo last point', self.canvas.undoLastPoint, shortcuts['undo_last_point'],
            'undo', 'Undo last drawn point', enable_condition=lambda: self.canvas.drawing()
        )
        selectAll = action(
            'Select All', self.labelList.select_and_check_all, shortcuts['select_all'],
            'select_all', 'Select all polygon labels', enabled=True
        )
        invertSelection = action('&Invert Selection', self.labelList.invertSelection,
                                 shortcuts['invert_selection'],
                                 'invert_selection', 'Invert selection of polygon labels',
                                 enabled=True)

        color1 = action('Polygon &Line Color', self.chooseColor1,
                        shortcuts['edit_line_color'], 'color_line',
                        'Choose polygon line color')
        color2 = action('Polygon &Fill Color', self.chooseColor2,
                        shortcuts['edit_fill_color'], 'color',
                        'Choose polygon fill color')

        toggle_keep_prev_mode = action(
            'Keep Previous Annotation',
            self.toggleKeepPrevMode,
            shortcuts['toggle_keep_prev_mode'], None,
            'Toggle "keep pevious annotation" mode',
            checkable=True)
        toggle_keep_prev_mode.setChecked(Config.get('keep_prev'))

        edit_menu = (
            allCreateModes,
            (edit, copy, delete, paste),
            (undo, undoLastPoint, selectAll, invertSelection),
            (color1, color2),
            (toggle_keep_prev_mode,)
        )

        # Help menu
        help_ = action(
            '&Tutorial', self.tutorial, icon='help',
            tip='Show tutorial page'
        )

        # Canvas menu
        editMode = action(
            'Edit Polygons', self.setEditMode, shortcuts['edit_polygon'],
            'edit', 'Move and edit the selected polygons',
            enable_condition=lambda: not self.canvas.is_empty() and not self.canvas.editing()
        )
        addPoint = action(
            'Add Point to Edge', self.canvas.addPointToEdge, None,
            'edit', 'Add point to the nearest edge', enabled=False
        )
        self.canvas.edgeSelected.connect(addPoint.setEnabled)

        # Zoom (for toolbar)
        zoom = QtWidgets.QWidgetAction(self)
        zoom.setDefaultWidget(self.zoomWidget)

        # Menus actions
        utils.addActions(self.menus.file, *file_menu)
        utils.addActions(self.menus.help, (help_,))
        utils.addActions(self.menus.view, *view_menu)
        utils.addActions(self.menus.edit, allCreateModes, *edit_menu)

        # Custom context menu for the canvas widget:
        utils.addActions(
            self.canvas.menus[0],
            allCreateModes,
            (
                editMode,
                edit,
                copy,
                delete,
                undo,
                undoLastPoint,
                addPoint,
            )
        )
        utils.addActions(
            self.canvas.menus[1],
            (
                action('&Copy here', self.copyShape),
                action('&Move here', self.moveShape),
            )
        )

        # Menu buttons on Left
        utils.addActions(
            self.tools,
            (
                open_, opendir, openNextImg, openPrevImg,
                save, deleteFile
            ),
            (
                allCreateModes[0], editMode,
                copy, delete, undo
            ),
            (
                zoomIn, zoom, zoomOut,
                fitWindow, fitWidth
            )
        )

        # Labellist menu
        utils.addActions(
            self.menus.labelList,
            (edit, delete, invertSelection, selectAll)
        )
