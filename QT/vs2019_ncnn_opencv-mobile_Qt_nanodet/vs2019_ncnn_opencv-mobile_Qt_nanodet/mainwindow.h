#pragma once

#include <QtWidgets/QMainWindow>
#include "ui_mainwindow.h"

#include <QFileDialog>

#include "net.h"
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <vector>
#include <iostream>
#include <math.h>
#include <fstream>

#include "openpose.h"
#include "lipjppt.h"
#include "densepose.h"
#include "ctnet.h"

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = Q_NULLPTR);

private:
    inline cv::Mat QImageToMat(const QImage& image);
    inline QImage MatToQImage(const cv::Mat& mat);
    cv::Mat letter256(cv::Mat in);

private slots:
    void on_openReferenceBtn_clicked();
    void on_openSourceBtn_clicked();
    void on_goBtn_clicked();
    void on_saveBtn_clicked();

private:
    Ui::MainWindowClass ui;

    OpenPose openpose;
    LIPJPPNet lipjpp;
    DensePose densepose;
    CTNet ct;
    
    cv::Mat source, reference, fake;
};
