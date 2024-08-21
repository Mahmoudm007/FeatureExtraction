#ifndef FILE_H
#define FILE_H

#include <opencv2/opencv.hpp>
#include <chrono>
#include <vector>
#include <cmath>
#include "template_matching.h"


cv::Mat applyGaussianBlur(const cv::Mat& input, double sigma);
void filterKeypoints(std::vector<cv::KeyPoint>& keypoints, const cv::Mat& image, double response_threshold);
void resizeImage(const cv::Mat& input, cv::Mat& output, double scale_factor);
void buildImagePyramid(const cv::Mat& image, std::vector<cv::Mat>& pyramid, int levels, double scale_factor);
void detectKeypoints(const std::vector<cv::Mat>& pyramid, std::vector<cv::KeyPoint>& keypoints, double response_threshold);
void cartToPolarCustom(const cv::Mat& grad_x, const cv::Mat& grad_y, cv::Mat& mag, cv::Mat& angle, bool angleInDegrees);
void computeDescriptors(const cv::Mat& image, const std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors);
int hammingDistance(const cv::Mat& desc1, const cv::Mat& desc2);
double computeSSD(const cv::Mat& desc1, const cv::Mat& desc2);
double computeNCC(const cv::Mat& desc1, const cv::Mat& desc2);
void matchDescriptors(const cv::Mat& descriptors1, const cv::Mat& descriptors2, std::vector<cv::DMatch>& matches, std::string method);
void drawMatchesFeatures(const cv::Mat& image1, const std::vector<cv::KeyPoint>& keypoints1,
    const cv::Mat& image2, const std::vector<cv::KeyPoint>& keypoints2,
    const std::vector<cv::DMatch>& matches, cv::Mat& output);
void drawKeypoints(const cv::Mat& image, const std::vector<cv::KeyPoint>& keypoints, cv::Mat& output);
cv::Mat computeKeypoints(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors, int levels, double scale_factor, double sigma, double response_threshold);
cv::Mat sift(const cv::Mat& image1, const cv::Mat& image2, SimilarityMetric metric, int levels, double scale_factor, double sigma, double response_threshold);

#endif // FILE_H
