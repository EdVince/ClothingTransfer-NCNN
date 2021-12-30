#pragma execution_character_set("utf-8")

#include "mainwindow.h"

#include<QDebug>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
{
    ui.setupUi(this);
    setWindowTitle("·þ×°Ç¨ÒÆ/ÐéÄâÊÔ´©(https://github.com/EdVince)");
}

void MainWindow::on_openReferenceBtn_clicked()
{
    QString fileName = QFileDialog::getOpenFileName(this, tr("open image file"), "./", tr("Image files(*.bmp *.jpg *.png);;All files (*.*)"));
    if (!fileName.isEmpty()) {
        std::string path = fileName.toLocal8Bit().toStdString();

        reference = cv::imread(path);
        cv::cvtColor(reference, reference, cv::COLOR_BGR2RGB);
        reference = letter256(reference);

        QPixmap pixmap;
        pixmap = pixmap.fromImage(MatToQImage(reference));
        pixmap = pixmap.scaled(ui.showReference->width(), ui.showReference->height(), Qt::KeepAspectRatio, Qt::SmoothTransformation);
        ui.showReference->setPixmap(pixmap);
    }
}

void MainWindow::on_openSourceBtn_clicked()
{
    QString fileName = QFileDialog::getOpenFileName(this, tr("open image file"), "./", tr("Image files(*.bmp *.jpg *.png);;All files (*.*)"));
    if (!fileName.isEmpty()) {
        std::string path = fileName.toLocal8Bit().toStdString();

        source = cv::imread(path);
        cv::cvtColor(source, source, cv::COLOR_BGR2RGB);
        source = letter256(source);

        QPixmap pixmap;
        pixmap = pixmap.fromImage(MatToQImage(source));
        pixmap = pixmap.scaled(ui.showSource->width(), ui.showSource->height(), Qt::KeepAspectRatio, Qt::SmoothTransformation);
        ui.showSource->setPixmap(pixmap);
    }
}

void MainWindow::on_goBtn_clicked()
{
    if (!source.empty() && !reference.empty()) {
        // read image
        cv::Mat src_rgb = source.clone();
        std::vector<cv::Mat> src_candidate_subset = openpose.call(src_rgb);
        cv::Mat src_seg = lipjpp.call(src_rgb);
        cv::Mat src_iuv = densepose.call(src_rgb);

        cv::Mat dst_rgb = reference.clone();
        std::vector<cv::Mat> dst_candidate_subset = openpose.call(dst_rgb);
        cv::Mat dst_seg = lipjpp.call(dst_rgb);

        cv::Mat src_candidate = src_candidate_subset[0];
        cv::Mat src_subset = src_candidate_subset[1];
        cv::Mat dst_candidate = dst_candidate_subset[0];
        cv::Mat dst_subset = dst_candidate_subset[1];
        fake = ct.call(src_rgb, dst_rgb, src_candidate, src_subset, dst_candidate, dst_subset, src_seg, dst_seg, src_iuv);

        QPixmap pixmap;
        pixmap = pixmap.fromImage(MatToQImage(fake));
        pixmap = pixmap.scaled(ui.showFake->width(), ui.showFake->height(), Qt::KeepAspectRatio, Qt::SmoothTransformation);
        ui.showFake->setPixmap(pixmap);
    }
}

void MainWindow::on_saveBtn_clicked()
{
    if (!fake.empty()) {
        QString fileName = QFileDialog::getSaveFileName(this, tr("save image file"), "./", tr("Image files(*.png)"));
        if (!fileName.isEmpty()) {
            std::string path = fileName.toLocal8Bit().toStdString();
            cv::Mat fakeRGB;
            cv::cvtColor(fake, fakeRGB, cv::COLOR_RGB2BGR);
            cv::imwrite(path, fakeRGB);
        }
    }
}

cv::Mat MainWindow::letter256(cv::Mat in)
{
    int height = in.rows, width = in.cols;
    int length = std::max(height, width);
    int hpad = length - height;
    int wpad = length - width;
    cv::Mat rectIn;
    copyMakeBorder(in, rectIn, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255));
    cv::Mat res;
    cv::resize(rectIn, res, cv::Size(256, 256));

    return res;
}

inline cv::Mat MainWindow::QImageToMat(const QImage& image)
{
    QImage swapped = image;
    if (image.format() == QImage::Format_RGB32) {
        swapped = swapped.convertToFormat(QImage::Format_RGB888);
    }

    return cv::Mat(swapped.height(), swapped.width(),
        CV_8UC3,
        const_cast<uchar*>(swapped.bits()),
        static_cast<size_t>(swapped.bytesPerLine())
    ).clone();
}

inline QImage MainWindow::MatToQImage(const cv::Mat& mat)
{
    return QImage(mat.data,
        mat.cols, mat.rows,
        static_cast<int>(mat.step),
        QImage::Format_RGB888);
}