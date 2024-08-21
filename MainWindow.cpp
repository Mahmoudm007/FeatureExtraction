#include "MainWindow.h"
#include "harris.h"
#include <QFileDialog> // Include necessary header
#include "sift.h"
#include "eigen.h"
#include "template_matching.h"

//cv::Mat img = cv::imread("orig.jpg");
//cv::Mat templ = cv::imread("temp.jpg");

// Convert images to grayscale
cv::Mat imgGray, templGray;
cv::Mat img_org, temp_org;




cv::Mat image;
cv::Mat gray_image;
cv::Mat siftOutput;
QString fileName;


cv::Mat siftinputImageMat2;
QString imgPathSift2;
QImage imj2;

cv::Mat siftinputImageMat1;
QString imgPathSift1;
QImage imj1;
float thresholdSliderValue;


cv::Mat harrisImage;
cv::Mat harrisImageGray;
QString imgPathHarris;

Mat match;
QString matchPath;

QString tempPath;
Mat temp;
std::string match_type = "SSD";

cv::Mat img_sift1;
QString img_sift1_path;
cv::Mat img_sift2;
QString img_sift2_path;

int thresholdValue;
int scaleValue;
int levelValue; 
int sigmaValue;




Mat img_key;
QString img_key_path;


MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindowClass())
{
    ui->setupUi(this);
    //connect(ui->browse, &QPushButton::clicked, this, &MainWindow::openImageDialog);
    //connect(ui->apply_corner_det, &QPushButton::clicked, this, &MainWindow::harris);    
    //connect(ui->apply_lambda, &QPushButton::clicked, this, &MainWindow::Eigen);


    connect(ui->apply_corner_det_2, &QPushButton::clicked, this, &MainWindow::harris);
    connect(ui->apply_lambda_2, &QPushButton::clicked, this, &MainWindow::Eigen);
    connect(ui->harrisBrowse_2, &QPushButton::clicked, this, &MainWindow::on_UploadeHarrisImage_clicked);




    connect(ui->match_browse, &QPushButton::clicked, this, &MainWindow::on_UploadeMatchImage_clicked);
    connect(ui->temp_browse, &QPushButton::clicked, this, &MainWindow::on_UploadeTempImage_clicked);
    connect(ui->match_apply, &QPushButton::clicked, this, &MainWindow::on_MatchApply_clicked);
    connect(ui->match_type, &QComboBox::currentTextChanged,this, &MainWindow::on_MatchType_changed);

    //         sift_page  
    connect(ui->sift_img1_btn, &QPushButton::clicked, this, &MainWindow::on_UploadeSiftImage1_clicked);
    connect(ui->sift_img2_btn, &QPushButton::clicked, this, &MainWindow::on_UploadeSiftImage2_clicked);
    connect(ui->apply_sift, &QPushButton::clicked, this, &MainWindow::Input_sift);



    connect(ui->in_key_btn, &QPushButton::clicked, this, &MainWindow::browse_key_points);
    connect(ui->app_key_btn, &QPushButton::clicked, this, &MainWindow::Input_key_points);

    
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::UploadImage(cv::Mat& imageMat, bool grayScale, QString& imgPath)
{
    // Open a file dialog to select an image file
    QString fileName = QFileDialog::getOpenFileName(this,
        tr("Open Image"), "", tr("Image Files (*.png *.jpg *.jpeg *.bmp)"));
    if (fileName.isEmpty()) return;  // User cancelled the dialog

    // Save the image path
    imgPath = fileName;

    // Load the image using cv::imread
    imageMat = cv::imread(fileName.toStdString());
    if (imageMat.empty()) {
        qDebug() << "Failed to load image from file: " << fileName;
        return;
    }

    // Optionally convert to grayscale
    if (grayScale && imageMat.channels() > 1) {
        cv::cvtColor(imageMat, imageMat, cv::COLOR_BGR2GRAY);
    }
}

void MainWindow::displayOutputImage(const cv::Mat& image, QLabel* label)
{
    if (!image.empty()) {
        cv::Mat displayImage;

        // Convert grayscale image to RGB for displaying
        if (image.channels() == 1) {
            cv::cvtColor(image, displayImage, cv::COLOR_GRAY2RGB);
        }
        else {
            displayImage = image.clone();
        }

        // Convert the OpenCV Mat image to QImage
        QImage qImage(displayImage.data, displayImage.cols, displayImage.rows, displayImage.step, QImage::Format_RGB888);
        if (qImage.isNull()) {
            std::cerr << "Error: Unable to convert image to QImage." << std::endl;
            return;
        }

        // Display the image in the specified QLabel
        QPixmap pixmap = QPixmap::fromImage(qImage);
        label->setPixmap(pixmap.scaled(label->size(), Qt::KeepAspectRatio));
        label->setAlignment(Qt::AlignCenter);
    }
    else {
        std::cerr << "Error: Image is empty." << std::endl;
    }
}
void MainWindow::displayImageInLabel(const cv::Mat& image, QLabel* label)
{
    if (!image.empty()) {
        // Convert the OpenCV Mat image to QImage with the same color format
        QImage qImage(image.data, image.cols, image.rows, image.step, QImage::Format_BGR888); // Assuming OpenCV loads in BGR format
        if (qImage.isNull()) {
            std::cerr << "Error: Unable to convert image to QImage." << std::endl;
            return;
        }

        // Display the image in the specified QLabel
        QPixmap pixmap = QPixmap::fromImage(qImage);
        label->setPixmap(pixmap.scaled(label->size(), Qt::KeepAspectRatio));
        label->setAlignment(Qt::AlignCenter);
    }
}










void MainWindow::on_UploadeHarrisImage_clicked()
{

    UploadImage(harrisImage, false, imgPathHarris);
    if (harrisImage.empty()) return;
    // UploadImage(harrisImageGray, true, imgPathHarris);
    cv::cvtColor(harrisImage, harrisImageGray, cv::COLOR_BGR2GRAY);
    displayImageInLabel(harrisImage, ui->input_color_image_corner_detection_2);
    displayOutputImage(harrisImageGray, ui->input_gray_image_corner_detection_2);
}

void MainWindow::on_UploadeMatchImage_clicked()
{
    UploadImage(match, false, matchPath);
    // if(match.data.empty()) return;
    match = loadImage(matchPath.toStdString());
    //cvtColor(match.data, match.data, cv::COLOR_BGR2GRAY);
    displayImageInLabel(match, ui->match_image);
    // displayOutputImage(harrisImageGray , ui->input_gray_image_corner_detection_2);
}


void MainWindow::on_UploadeTempImage_clicked()
{

    UploadImage(temp, false, tempPath);
    // if(temp.data.empty()) return;
    temp = loadImage(tempPath.toStdString());
    //cvtColor(temp.data, temp.data, cv::COLOR_BGR2GRAY);
    displayImageInLabel(temp, ui->temp_image);

    // displayOutputImage(harrisImageGray , ui->input_gray_image_corner_detection_2);
}



void MainWindow::Eigen()
{
    if (!harrisImage.empty()) {
        cv::Mat eigen_img = process_Image_eigen(harrisImage);
        displayImageInLabel(eigen_img, ui->corner_detected_output_2);
    }
    else {
        std::cerr << "Error: Image is empty." << std::endl;
    }
}


void MainWindow::harris()
{
    if (!harrisImage.empty()) {
        cv::Mat harris_img = processImage(harrisImage);
        displayImageInLabel(harris_img, ui->corner_detected_output_2);
    }
    else {
        std::cerr << "Error: Image is empty." << std::endl;
    }
}

void MainWindow::on_MatchType_changed(const QString& text) {
    match_type = text.toStdString();
    std::cout << match_type;
}

void MainWindow::on_MatchApply_clicked() {
    cv::Mat matched;
    if (match_type == "ssd") {


        cvtColor(matched, matched, cv::COLOR_BGR2GRAY);

        cvtColor(match, imgGray, cv::COLOR_BGR2GRAY);
        cvtColor(temp, templGray, cv::COLOR_BGR2GRAY);

        Rect roi = ssdMatch(imgGray, templGray);

        // Draw the rectangle on the color image
        rectangle(matched, roi, cv::Scalar(0, 255, 0), 2);


        //matched = findTemplate(match, temp, SimilarityMetric::SSD);
        displayImageInLabel(matched, ui->match_output);
    }
    else {
        matched = findTemplate(match, temp, SimilarityMetric::NCC);
        displayImageInLabel(matched, ui->match_output);
    }
}




// ------------------------------------------------------------------------------ sift

void MainWindow::on_UploadeSiftImage1_clicked()
{

    UploadImage(img_sift1, false, img_sift1_path);
    // if(temp.data.empty()) return;
    temp = loadImage(img_sift1_path.toStdString());
    displayImageInLabel(img_sift1, ui->sift_img1_label);
    // displayOutputImage(harrisImageGray , ui->input_gray_image_corner_detection_2);
}

void MainWindow::on_UploadeSiftImage2_clicked()
{

    UploadImage(img_sift2, false, img_sift2_path);
    // if(temp.data.empty()) return;
    temp = loadImage(img_sift2_path.toStdString());
    displayImageInLabel(img_sift2, ui->sift_img2_label);
    // displayOutputImage(harrisImageGray , ui->input_gray_image_corner_detection_2);
}


void MainWindow::Input_sift()
{
    // Get the text from the line edits and combo box
    QString inputText_t = ui->threshold_sift_input->text();
    QString inputText_s = ui->scale_factor_sift_input->text();
    QString inputText_l = ui->level_sift_input->text();
    QString inputText_si = ui->sigma_sift_input->text();

    // Convert the text to integers
    bool ok;
    double thresholdValue = inputText_t.toDouble(&ok);
    double scaleValue = inputText_s.toDouble(&ok);
    double levelValue = inputText_l.toDouble(&ok);
    double sigmaValue = inputText_si.toDouble(&ok);

    if (!ok) {
        qDebug() << "Invalid input. Please enter valid integers.";
        // Handle invalid input
        return;
    }

    // Get the selected item from the combo box
    QString selectedMetric = ui->sift_type->currentText();
    SimilarityMetric metric;

    // Map the selected item to the enum value
    if (selectedMetric == "ssd") {

        metric = SimilarityMetric::SSD;
    }
    else if (selectedMetric == "ncc") {
        metric = SimilarityMetric::NCC;
    }
    else {
        qDebug() << "Invalid metric selection.";
        // Handle invalid metric selection
        return;
    }

    // Call the sift function with the parameters
    cv::Mat res = sift(img_sift1, img_sift2, metric, levelValue, scaleValue, sigmaValue, thresholdValue);
    displayImageInLabel(res, ui->output_sift_img);
}

//    ----------------------Key_points
void MainWindow::browse_key_points() 
{

    UploadImage(img_key, false, img_key_path);
    // if(temp.data.empty()) return;
    temp = loadImage(img_key_path.toStdString());
    displayImageInLabel(img_key, ui->in_key);
    // displayOutputImage(harrisImageGray , ui->input_gray_image_corner_detection_2);
}

void MainWindow::Input_key_points()
{


    //QString inputText_t = ui->th_l->text();
    QString inputText_s = ui->sc_l->text();
    QString inputText_l = ui->lev_l->text();
    QString inputText_si = ui->su_l->text();


    bool ok;
    //double thresholdValue = inputText_t.toDouble(&ok);
    double scaleValue = inputText_s.toDouble(&ok);
    double levelValue = inputText_l.toDouble(&ok);
    double sigmaValue = inputText_si.toDouble(&ok);

    if (!ok) {
        qDebug() << "Invalid input. Please enter valid integers.";
        // Handle invalid input
        return;
    }

    //Mat computeKeypoints(const Mat & image, std::vector<KeyPoint>&keypoints, Mat & descriptors, int levels, double scale_factor, double sigma, double response_threshold) {

    // Call the sift function with the parameters
    std::vector<KeyPoint> keypoints;
         Mat descriptors;

    cv::Mat res = computeKeypoints(img_key, keypoints, descriptors, levelValue, scaleValue, sigmaValue, 0.001);
    displayImageInLabel(res, ui->out_key);
}