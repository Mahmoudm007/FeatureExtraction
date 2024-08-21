#include <iostream>
#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <chrono>
#include "template_matching.h"
// Define Image structure to represent grayscale images
// struct Image {
//     cv::Mat data;
// };

// // Enum for similarity metric
// enum class SimilarityMetric {
//     SSD,
//     NCC
// };

// Function to load image using OpenCV
cv::Mat loadImage(const std::string& filename) {
    cv::Mat img = cv::imread(filename);
    if (img.empty()) {
        std::cerr << "Error: Could not load image " << filename << std::endl;
        exit(1);
    }
    cv::Mat image;
    image = img;
    return image;
}

// Function to calculate Sum of Squared Differences (SSD) between two images
cv::Rect ssdMatch(const cv::Mat& img, const cv::Mat& templ) {
    int resultCols = img.cols - templ.cols + 1;
    int resultRows = img.rows - templ.rows + 1;
    cv::Mat output(resultRows, resultCols, CV_32S);

    for (int i = 0; i < resultRows; ++i) {
        for (int j = 0; j < resultCols; ++j) {
            cv::Mat roi(img, cv::Rect(j, i, templ.cols, templ.rows));
            cv::Mat diff;
            cv::absdiff(roi, templ, diff);
            cv::Mat resultImg = diff.mul(diff);
            output.at<int>(i, j) = cv::sum(resultImg)[0];
        }
    }

    cv::Point minLoc;
    cv::minMaxLoc(output, nullptr, nullptr, &minLoc);
    return cv::Rect(minLoc.x, minLoc.y, templ.cols, templ.rows);
}


// Function to calculate Normalized Cross-Correlation (NCC) between two images
double calculateNCC(const cv::Mat& img1, const cv::Mat& img2) {
    // Ensure images have the same size
    assert(img1.size() == img2.size());

    double mean1 = mean(img1)[0];
    double mean2 = mean(img2)[0];

    double numerator = 0.0;
    double denominator_img1 = 0.0;
    double denominator_img2 = 0.0;

    for (int y = 0; y < img1.rows; ++y) {
        for (int x = 0; x < img1.cols; ++x) {
            double diff1 = img1.at<uchar>(y, x) - mean1;
            double diff2 = img2.at<uchar>(y, x) - mean2;
            numerator += diff1 * diff2;
            denominator_img1 += diff1 * diff1;
            denominator_img2 += diff2 * diff2;
        }
    }

    double ncc = numerator / sqrt(denominator_img1 * denominator_img2);
    return ncc;
}

// Function to calculate similarity between original image and template at a given position
double calculateSimilarity(const Mat& original, const Mat& templateImage, int x, int y, SimilarityMetric metric, double& elapsedTimeMicro, double& elapsedTimeNano) {
    double similarityScore = 0.0;

    // Extract region of interest (ROI) from original image
    cv::Mat roi(original, cv::Rect(x, y, templateImage.cols, templateImage.rows));

    // Measure elapsed time for the operation in microseconds
    auto startTimeMicro = std::chrono::high_resolution_clock::now();

    // Calculate similarity based on chosen metric          ssdMatch(const cv::Mat& img, const cv::Mat& templ)

        similarityScore = calculateNCC(roi, templateImage);

    

    auto endTimeMicro = std::chrono::high_resolution_clock::now();
    elapsedTimeMicro = std::chrono::duration_cast<std::chrono::microseconds>(endTimeMicro - startTimeMicro).count();

    // Measure elapsed time for the operation in nanoseconds
    auto startTimeNano = std::chrono::high_resolution_clock::now();


   
        similarityScore = calculateNCC(roi, templateImage);
     
    auto endTimeNano = std::chrono::high_resolution_clock::now();
    elapsedTimeNano = std::chrono::duration_cast<std::chrono::nanoseconds>(endTimeNano - startTimeNano).count();

    return similarityScore;
}

// Function to find template in original image
cv::Mat findTemplate(const Mat& original, const Mat& templateImage, SimilarityMetric metric) {
    double maxSimilarityScore = std::numeric_limits<double>::min();
    int bestMatchX = 0;
    int bestMatchY = 0;

    double elapsedTimeMicro = 0.0; // Variable to store elapsed time in microseconds
    double elapsedTimeNano = 0.0; // Variable to store elapsed time in nanoseconds

    // Iterate over each pixel in the original image
    for (int y = 0; y <= original.rows - templateImage.rows; ++y) {
        for (int x = 0; x <= original.cols - templateImage.cols; ++x) {
            // Calculate similarity score between original image and template
            double similarityScore = calculateSimilarity(original, templateImage, x, y, metric, elapsedTimeMicro, elapsedTimeNano);

            // Update maximum similarity score and position
            if (similarityScore > maxSimilarityScore) {
                maxSimilarityScore = similarityScore;
                bestMatchX = x;
                bestMatchY = y;
            }
        }
    }

    std::string metricName;

        metricName = "NCC";


    std::cout << "Elapsed time (" << metricName << ") in microseconds: " << elapsedTimeMicro << std::endl;
    std::cout << "Elapsed time (" << metricName << ") in nanoseconds: " << elapsedTimeNano << std::endl;

    // Draw rectangle around matched region
    cv::rectangle(original, cv::Point(bestMatchX, bestMatchY),
        cv::Point(bestMatchX + templateImage.cols, bestMatchY + templateImage.rows),
        cv::Scalar(255, 0, 0), 2);

    return original;

    // // Display result
    // cv::imshow("Result", original.data);
    // cv::waitKey(0);
}

// int main() {
//     // Load original and template images using OpenCV
//     Image original = loadImage("orig.jpg");
//     Image templateImage = loadImage("temp.jpg");

//     //// Find and draw template in original image using SSD
//     std::cout << "Using Sum of Squared Differences (SSD)" << std::endl;
//     findTemplate(original, templateImage, SimilarityMetric::SSD);

//     // Find and draw template in original image using NCC
//     std::cout << "Using Normalized Cross-Correlation (NCC)" << std::endl;
//     findTemplate(original, templateImage, SimilarityMetric::NCC);

//     return 0;
// }
