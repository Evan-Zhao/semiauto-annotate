#ifndef LABELDIALOGBOX_H
#define LABELDIALOGBOX_H

#include <QWidget>

namespace Ui {
class LabelDialogBox;
}

class LabelDialogBox : public QWidget
{
    Q_OBJECT

public:
    explicit LabelDialogBox(QWidget *parent = nullptr);
    ~LabelDialogBox();

private:
    Ui::LabelDialogBox *ui;
};

#endif // LABELDIALOGBOX_H
