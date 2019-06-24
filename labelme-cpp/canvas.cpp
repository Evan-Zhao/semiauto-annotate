#include "canvas.h"
#include "ui_canvas.h"

Canvas::Canvas(QWidget *parent, float epsilon,
               std::unordered_map<std::string, std::string> labelColor)
    : QWidget(parent),
      ui(new Ui::Canvas),
      epsilon(epsilon),
      labelColor(labelColor) {
    ui->setupUi(this);

    this->line = new Shape(nullptr, &this->lineColor, nullptr);
    // Set widget options.
    this->setMouseTracking(true);
    this->setFocusPolicy(Qt::WheelFocus);
}

Canvas::~Canvas() { delete ui; }
