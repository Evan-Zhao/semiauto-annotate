#ifndef SHAPE_H
#define SHAPE_H

#include <cstdint>
#include <functional>
#include <unordered_map>
#include <vector>

#include <QColor>
#include <QPointF>

class Shape {
    enum class highlightMode { MOVE_VERTEX = 0, NEAR_VERTEX = 1 };
    enum class highlightShape { P_SQUARE = 0, P_ROUND = 1 };

public:
    Shape(const std::string *label, const QColor *line_color,
          const std::string *shape_type);

private:
    std::string label;
    std::vector<QPointF> points;
    bool fill = false, selected = false, _closed = false;
    std::string shape_type;
    highlightMode _highlightMode = highlightMode::NEAR_VERTEX;
    std::unordered_map<highlightMode, std::tuple<float, highlightShape>>
        _highlightSettings = {
            {highlightMode::NEAR_VERTEX, {4, highlightShape::P_ROUND}},
            {highlightMode::MOVE_VERTEX, {1.5, highlightShape::P_SQUARE}},
    };

    def paint(self, painter):
        if not self.points:
            return
        color = self.line_color
        pen = QtGui.QPen(color)
        # Try using integer sizes for smoother drawing(?)
        pen.setWidth(max(1, int(round(2.0 / self.scale))))
        painter.setPen(pen)

        line_path = QtGui.QPainterPath()
        vrtx_path = QtGui.QPainterPath()

        if self.shape_type == 'rectangle':
            assert len(self.points) in [1, 2]
            if len(self.points) == 2:
                rectangle = self.getRectFromLine(*self.points)
                line_path.addRect(rectangle)
            for i in range(len(self.points)):
                self.drawVertex(vrtx_path, i)
        elif self.shape_type == "circle":
            assert len(self.points) in [1, 2]
            if len(self.points) == 2:
                rectangle = self.getCircleRectFromLine(self.points)
                line_path.addEllipse(rectangle)
            for i in range(len(self.points)):
                self.drawVertex(vrtx_path, i)
        elif self.shape_type == "linestrip":
            line_path.moveTo(self.points[0])
            for i, p in enumerate(self.points):
                line_path.lineTo(p)
                self.drawVertex(vrtx_path, i)
        elif self.shape_type == 'curve':
            for i in range(len(self.points)):
                self.drawVertex(vrtx_path, i)
            # Paint Bezier curve across given points.
            refined_points = BezierB(self.points).smooth()
            line_path.moveTo(self.points[0])
            for p in refined_points:
                line_path.lineTo(p)
        else:
            line_path.moveTo(self.points[0])
            # Uncommenting the following line will draw 2 paths
            # for the 1st vertex, and make it non-filled, which
            # may be desirable.
            # self.drawVertex(vrtx_path, 0)

            for i, p in enumerate(self.points):
                line_path.lineTo(p)
                self.drawVertex(vrtx_path, i)
            if self.isClosed():
                line_path.lineTo(self.points[0])

        painter.drawPath(line_path)
        painter.drawPath(vrtx_path)
        painter.fillPath(vrtx_path, self.vertex_fill_color)
        if self.fill:
            color = self.fill_color
            painter.fillPath(line_path, color)

};

namespace std {
template <>
struct hash<Shape> {
    std::size_t operator()(const Shape &s) const { return 0; }
};
}  // namespace std

#endif  // SHAPE_H
