#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

Mat process_Image_eigen(const Mat& colorImage) {
    if (colorImage.empty()) {
        cerr << "Error: Input image is empty." << endl;
        return Mat();
    }

    Mat grayImage;
    cvtColor(colorImage, grayImage, COLOR_BGR2GRAY);
    GaussianBlur(grayImage, grayImage, Size(5, 5), 0, 0);

    // Compute Ix & Iy using Scharr operator
    Mat dx, dy;
    Scharr(grayImage, dx, CV_32F, 1, 0);
    Scharr(grayImage, dy, CV_32F, 0, 1);

    //Create H Matrix
    Mat h11, h12, h21, h22;
    multiply(dx, dx, h11);
    multiply(dy, dy, h22);
    multiply(dx, dy, h12);
    multiply(dy, dx, h21);

    // Compute Lambda
    Mat lambda_plus(h11.size(), h11.type());
    Mat lambda_minus(h11.size(), h11.type());
    Mat temp, temp2, temp3;
    multiply(h21, h12, temp);
    temp *= 4;
    subtract(h11, h22, temp2);
    pow(temp2, 2, temp2);
    add(temp, temp2, temp2);
    sqrt(temp2, temp2);
    add(h11, h22, temp3);
    add(temp3, temp2, lambda_plus);
    subtract(temp3, temp2, lambda_minus);
    lambda_plus *= 0.5;
    lambda_minus *= 0.5;

    // Find features
    double threshold = 0.0; // Adjust threshold as needed
    vector<Point> features;
    Mat normalized_lambda;
    normalize(lambda_minus, normalized_lambda, 0, 255, NORM_MINMAX, CV_8U);
    for (int y = 1; y < lambda_minus.rows - 1; ++y) {
        for (int x = 1; x < lambda_minus.cols - 1; ++x) {
            double value = lambda_minus.at<float>(y, x);
            if (value > threshold) {
                if (value > lambda_minus.at<float>(y - 1, x - 1) &&
                    value > lambda_minus.at<float>(y - 1, x) &&
                    value > lambda_minus.at<float>(y - 1, x + 1) &&
                    value > lambda_minus.at<float>(y, x - 1) &&
                    value > lambda_minus.at<float>(y, x + 1) &&
                    value > lambda_minus.at<float>(y + 1, x - 1) &&
                    value > lambda_minus.at<float>(y + 1, x) &&
                    value > lambda_minus.at<float>(y + 1, x + 1)) {
                    features.push_back(Point(x, y));
                }
            }
        }
    }

    // Check if any features were found
    if (features.empty()) {
        cout << "No features found." << endl;
    }
    else {
        cout << "Number of features found: " << features.size() << endl;
    }

    // Display features on the image
    Mat resultImage = colorImage.clone();
    for (const auto& point : features) {
        circle(resultImage, point, 5, Scalar(0, 255, 0), 2);
    }

    return resultImage;
}
//
//int main() {
//    // Load the input image
//    Mat colorImage = imread("C:/Users/memaa/Documents/Computer Vision/TEST/images.png");
//    Mat processedImage = processImage(colorImage);
//    if (!processedImage.empty()) {
//        imshow("Processed Image", processedImage);
//        waitKey(0);
//    }
//    return 0;
//}
