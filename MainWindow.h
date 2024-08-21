#pragma once

#include <QtWidgets/QMainWindow>
#include "ui_MainWindow.h"
#include <QMainWindow>
#include <QLabel>
#include <QFileDialog>
#include <opencv2/opencv.hpp> 

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindowClass; };
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

    //void openImageDialog();

    //void update_Threshold(int value);


    //void displayOutputImage(const cv::Mat& image, QLabel* label);

    //void displayImageInLabel(const cv::Mat& image, QLabel* label);
    //void on_UploadeHarrisImage_clicked();

    //void Eigen();



    //void harris();



    void UploadImage(cv::Mat& imageMat, bool grayScale, QString& imgPath);
    void on_UploadeSiftImage1_clicked();
    void on_UploadeSiftImage2_clicked();
    void onThresholdSiftInputFinished();
    void Input_sift();



    void displayImageInLabel(const cv::Mat& image, QLabel* label);
    void displayOutputImage(const cv::Mat& image, QLabel* label);

    void browse_key_points();
    void Input_key_points();



    void Eigen();
    void harris();
    void on_UploadeHarrisImage_clicked();


    void on_UploadeMatchImage_clicked();
    void on_UploadeTempImage_clicked();
    void on_MatchApply_clicked();
    void on_MatchType_changed(const QString& text);


private:
    Ui::MainWindowClass *ui;
};
