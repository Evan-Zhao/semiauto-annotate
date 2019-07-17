import os.path as osp
from math import sqrt

import numpy as np
from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets

here = osp.dirname(osp.abspath(__file__))


def newIcon(icon):
    icons_dir = osp.join(here, '../icons')
    return QtGui.QIcon(osp.join(':/', icons_dir, '%s.png' % icon))


def newButton(text, icon=None, slot=None):
    b = QtWidgets.QPushButton(text)
    if icon is not None:
        b.setIcon(newIcon(icon))
    if slot is not None:
        b.clicked.connect(slot)
    return b


class BindingQAction(QtWidgets.QAction):
    def __init__(self, enable_condition, *args):
        super(BindingQAction, self).__init__(*args)
        self.enable_condition = enable_condition

    def set_condition(self, cond):
        self.enable_condition = cond

    def refresh(self):
        if self.enable_condition:
            super().setEnabled(self.enable_condition())

    def setEnabled(self, value):
        if self.enable_condition:
            result = self.enable_condition()
            print(f"Don't set a BindingQAction! Will refresh to {result} now.")
            self.refresh()
        else:
            super().setEnabled(value)


class ActionStorage(object):
    def __init__(self, parent):
        self.parent = parent
        self.actions = []

    def make_action(self, text, slot=None, shortcut=None, icon=None,
                    tip=None, checkable=False, enabled=True, enable_condition=None):
        """Create a new action and assign callbacks, shortcuts, etc."""
        a = BindingQAction(enable_condition, text, self.parent)
        if icon is not None:
            a.setIconText(text.replace(' ', '\n'))
            a.setIcon(newIcon(icon))
        if shortcut is not None:
            if isinstance(shortcut, (list, tuple)):
                a.setShortcuts(shortcut)
            else:
                a.setShortcut(shortcut)
        if tip is not None:
            a.setToolTip(tip)
            a.setStatusTip(tip)
        if slot is not None:
            a.triggered.connect(slot)
        if checkable:
            a.setCheckable(True)
        if enable_condition:
            a.refresh()
        else:
            a.setEnabled(enabled)
        self.actions.append(a)
        return a

    def make_group(self, *args, exclusive=False):
        group = QtWidgets.QActionGroup(self.parent)
        for action in args:
            group.addAction(action)
        group.setExclusive(exclusive)
        return group

    def refresh_all(self):
        for action in self.actions:
            action.refresh()


def addActions(widget, *actions):
    def add_actions_or_menus(widget_, actions_):
        for action_ in actions_:
            if isinstance(action_, QtWidgets.QMenu):
                widget_.addMenu(action_)
            else:
                widget_.addAction(action_)

    from typing import Iterable
    for action in actions:
        if issubclass(type(action), Iterable):
            add_actions_or_menus(widget, action)
        elif isinstance(action, QtWidgets.QActionGroup):
            widget.addActions(action.actions())
        widget.addSeparator()


def labelValidator():
    return QtGui.QRegExpValidator(QtCore.QRegExp(r'^[^ \t].+'), None)


def distance(p):
    return sqrt(p.x() * p.x() + p.y() * p.y())


def distancetoline(point, line):
    p1, p2 = line
    p1 = np.array([p1.x(), p1.y()])
    p2 = np.array([p2.x(), p2.y()])
    p3 = np.array([point.x(), point.y()])
    if np.dot((p3 - p1), (p2 - p1)) < 0:
        return np.linalg.norm(p3 - p1)
    if np.dot((p3 - p2), (p1 - p2)) < 0:
        return np.linalg.norm(p3 - p2)
    return np.linalg.norm(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)


def fmtShortcut(text):
    mod, key = text.split('+', 1)
    return '<b>%s</b>+<b>%s</b>' % (mod, key)
