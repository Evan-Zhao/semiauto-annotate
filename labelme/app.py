from qtpy.QtWidgets import QApplication


class Application(QApplication):
    def __init__(self, *args, **kwargs):
        super(Application, self).__init__(*args, **kwargs)
        self.main_window = None

    def set_main_window(self, window):
        self.main_window = window

    @staticmethod
    def get_main_window():
        return Application.instance().main_window
