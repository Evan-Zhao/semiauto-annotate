#ifndef CANVAS_H
#define CANVAS_H

#include <QMenu>
#include <QPainter>
#include <QWidget>
#include <QtCore>

#include <string>
#include <unordered_map>

#include "shape.h"

namespace Ui {
class Canvas;
}

class Canvas : public QWidget {
    Q_OBJECT
    enum class mode { CREATE = 0, EDIT = 1 };

    public:
    explicit Canvas(QWidget *parent, float epsilon,
                    std::unordered_map<std::string, std::string> labelColor);

    ~Canvas() override;

    private:
    Ui::Canvas *ui;

    const float epsilon;
    const std::unordered_map<std::string, std::string> labelColor;

    // polygon, rectangle, line, or point
    std::string _createMode = "polygon";
    bool _fill_drawing = false;

    mode mode = mode::EDIT;
    // save the selected shapes here
    std::vector<Shape> shapes, shapesBackups, selectedShapes,
        selectedShapesCopy;
    QColor lineColor = QColor(0, 0, 255);
    Shape *line, *hShape, *hVertex, *hEdge;
    QPointF prevPoint, prevMovePoint, imagePos;
    std::tuple<QPointF, QPointF> offsets;
    float scale = 1.0;
    QPixmap pixmap;
    bool _hideBackround = false, hideBackround = false, movingShape = false;
    QPainter _painter;
    int _cursor;
    /* Menus:
       0: right-click without selection and dragging of shapes
       1: right-click with selection and dragging of shapes */
    std::tuple<QMenu, QMenu> menus;
    std::unordered_map<Shape, bool> visible;
};

#endif  // CANVAS_H
