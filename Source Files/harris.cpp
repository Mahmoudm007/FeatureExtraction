#include "harris.h"

using namespace cv;
using namespace std;


void computeHarrisResponse(const Mat& dx, const Mat& dy, double lambda, Mat& harrisResponse) {
    Mat dx2, dy2, dxy;
    multiply(dx, dx, dx2);
    multiply(dy, dy, dy2);
    multiply(dx, dy, dxy);

    Mat trace = dx2 + dy2;
    Mat det = dx2.mul(dy2) - dxy.mul(dxy);

    harrisResponse = det - lambda * trace.mul(trace);
}

void findCornerPoints(const Mat& harrisResponse, vector<Point>& cornerPoints, double threshold) {
    Mat harrisNorm;
    normalize(harrisResponse, harrisNorm, 0, 255, NORM_MINMAX, CV_8U);

    for (int y = 1; y < harrisNorm.rows - 1; ++y) {
        for (int x = 1; x < harrisNorm.cols - 1; ++x) {
            uchar val = harrisNorm.at<uchar>(y, x);
            if (val > threshold &&
                val > harrisNorm.at<uchar>(y - 1, x - 1) &&
                val > harrisNorm.at<uchar>(y - 1, x) &&
                val > harrisNorm.at<uchar>(y - 1, x + 1) &&
                val > harrisNorm.at<uchar>(y, x - 1) &&
                val > harrisNorm.at<uchar>(y, x + 1) &&
                val > harrisNorm.at<uchar>(y + 1, x - 1) &&
                val > harrisNorm.at<uchar>(y + 1, x) &&
                val > harrisNorm.at<uchar>(y + 1, x + 1)) {
                cornerPoints.push_back(Point(x, y));
            }
        }
    }
}

void drawCornerPoints(Mat& image, const vector<Point>& cornerPoints, const Scalar& color, int radius, int thickness) {
    for (const Point& pt : cornerPoints) {
        circle(image, pt, radius, color, thickness);
    }
}

void computeGradients(const cv::Mat& inputImage, cv::Mat& gradX, cv::Mat& gradY) {
    Scharr(inputImage, gradX, CV_32F, 1, 0);
    Scharr(inputImage, gradY, CV_32F, 0, 1);
}


void computeStructureTensor(const cv::Mat& gradX, const cv::Mat& gradY, cv::Mat& M11, cv::Mat& M12, cv::Mat& M22) {
    M11 = gradX.mul(gradX);
    M12 = gradX.mul(gradY);
    M22 = gradY.mul(gradY);
}

void detectEdges(const cv::Mat& inputImage, cv::Mat& outputImage) {
    cv::Mat gradX, gradY, M11, M12, M22;
    computeGradients(inputImage, gradX, gradY);
    computeStructureTensor(gradX, gradY, M11, M12, M22);

    // Compute eigenvalues of M
    cv::Mat eigenvalues(inputImage.size(), CV_32FC2);
    cv::eigen(M11 + M22, eigenvalues);

    // Draw circles based on eigenvalues
    outputImage = inputImage.clone();
    for (int i = 0; i < eigenvalues.rows; ++i) {
        float lambda1 = eigenvalues.at<cv::Vec2f>(i)[0];
        float lambda2 = eigenvalues.at<cv::Vec2f>(i)[1];

        if (lambda1 > 100) { // Adjust threshold as needed
            cv::Point center(i % eigenvalues.cols, i / eigenvalues.cols);
            cv::circle(outputImage, center, 3, cv::Scalar(0, 0, 255), cv::FILLED); // Draw red circles
        }
    }
}

cv::Mat processImage(const cv::Mat& image)
{
    cv::Mat colorImage = image.clone();  // Make a copy to avoid modifying the original image
    cv::Mat grayImage;

    // Convert the image to grayscale
    cvtColor(colorImage, grayImage, cv::COLOR_BGR2GRAY);
    GaussianBlur(grayImage, grayImage, cv::Size(5, 5), 0, 0);

    // Compute gradients and other processing steps (no color space conversions needed here)
    cv::Mat dx, dy;
    computeGradients(grayImage, dx, dy);

    cv::Mat harrisResponse;
    computeHarrisResponse(dx, dy, 0.04, harrisResponse);

    std::vector<cv::Point> cornerPoints;
    findCornerPoints(harrisResponse, cornerPoints, 20.0);

    drawCornerPoints(colorImage, cornerPoints, cv::Scalar(0, 255, 0), 5, 2);

    return colorImage;
}


cv::Mat convertToGrayScale(const cv::Mat& input) {
    cv::Mat output(input.rows, input.cols, CV_8UC1);

    // Iterate through each pixel
    for (int y = 0; y < input.rows; ++y) {
        for (int x = 0; x < input.cols; ++x) {
            // Get the pixel value
            cv::Vec3b pixel = input.at<cv::Vec3b>(y, x);
            // Compute grayscale value using formula: Y = 0.299R + 0.587G + 0.114B
            unsigned char gray = (unsigned char)(0.299 * pixel[2] + 0.587 * pixel[1] + 0.114 * pixel[0]);
            // Set the grayscale value for all channels
            output.at<unsigned char>(y, x) = gray;
        }
    }

    return output;
}

