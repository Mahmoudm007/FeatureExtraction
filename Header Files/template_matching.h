#ifndef TEMPLATE_MATCHING_H
#define TEMPLATE_MATCHING_H
// Define Image structure to represent grayscale images
#include "opencv2/core/mat.hpp"
using namespace cv;
//cv::Mat data;

//struct Image {
//};

// Enum for similarity metric
typedef enum class SimilarityMetric {
    SSD,
    NCC
};

// Function to load image using OpenCV
cv::Mat loadImage(const std::string& filename);

// Function to calculate Sum of Squared Differences (SSD) between two images
//double calculateSSD(const cv::Mat& img1, const cv::Mat& img2);

// Function to calculate Normalized Cross-Correlation (NCC) between two images
double calculateNCC(const cv::Mat& img1, const cv::Mat& img2);

// Function to calculate similarity between original image and template at a given position
double calculateSimilarity(const Mat& original, const Mat& templateImage, int x, int y, SimilarityMetric metric, double& elapsedTimeMicro, double& elapsedTimeNano);

// Function to find template in original image
cv::Mat findTemplate(const Mat& original, const Mat& templateImage, SimilarityMetric metric);


cv::Rect ssdMatch(const cv::Mat& img, const cv::Mat& templ);


#endif // TEMPLATE_MATCHING_H
