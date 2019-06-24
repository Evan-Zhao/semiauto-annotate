#include "labeldialogbox.h"
#include "ui_labeldialogbox.h"

LabelDialogBox::LabelDialogBox(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::LabelDialogBox)
{
    ui->setupUi(this);
}

LabelDialogBox::~LabelDialogBox()
{
    delete ui;
}
