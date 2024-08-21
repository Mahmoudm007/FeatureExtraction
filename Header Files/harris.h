#ifndef HARRIS_H
#define HARRIS_H

#include <opencv2/opencv.hpp>
#include <vector>

void computeHarrisResponse(const cv::Mat& dx, const cv::Mat& dy, double lambda, cv::Mat& harrisResponse);
void findCornerPoints(const cv::Mat& harrisResponse, std::vector<cv::Point>& cornerPoints, double threshold);
void drawCornerPoints(cv::Mat& image, const std::vector<cv::Point>& cornerPoints, const cv::Scalar& color, int radius, int thickness);
void computeGradients(const cv::Mat& inputImage, cv::Mat& gradX, cv::Mat& gradY);
void computeStructureTensor(const cv::Mat& gradX, const cv::Mat& gradY, cv::Mat& M11, cv::Mat& M12, cv::Mat& M22);
void detectEdges(const cv::Mat& inputImage, cv::Mat& outputImage);
cv::Mat processImage(const cv::Mat& image);
cv::Mat convertToGrayScale(const cv::Mat& input);

#endif // HARRIS_H
