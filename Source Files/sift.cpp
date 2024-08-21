#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <vector>
#include <cmath>
#include "sift.h"
#include "template_matching.h"
using namespace cv;
using namespace std;
using namespace std::chrono; // Namespace for timing


Mat applyGaussianBlur(const Mat& input, double sigma) {
    // Ensure sigma is non-negative
    if (sigma < 0) {
        std::cerr << "Error: Sigma value must be non-negative." << std::endl;
        return Mat();
    }

    // Calculate kernel size based on sigma
    int kernel_size = cvRound(6 * sigma) + 1;

    // Create an empty kernel matrix
    Mat kernel(kernel_size, kernel_size, CV_64F);

    // Calculate constants
    double mean = kernel_size / 2;
    double scale = 1.0 / (sigma * sqrt(2 * CV_PI));

    // Fill the kernel values
    for (int i = 0; i < kernel_size; ++i) {
        for (int j = 0; j < kernel_size; ++j) {
            double x = i - mean;
            double y = j - mean;
            double value = scale * exp(-(x * x + y * y) / (2 * sigma * sigma));
            kernel.at<double>(i, j) = value;
        }
    }

    // Normalize the kernel
    normalize(kernel, kernel, 1, 0, NORM_L1);

    // Apply the convolution operation
    Mat blurred;
    filter2D(input, blurred, -1, kernel, Point(-1, -1), 0, BORDER_DEFAULT);

    return blurred;
}

void filterKeypoints(std::vector<KeyPoint>& keypoints, const Mat& image, double response_threshold) {
    std::vector<KeyPoint> filtered_keypoints;
    for (const auto& kp : keypoints) {
        // Check if the keypoint's response value is above the threshold
        if (kp.response >= response_threshold) {
            filtered_keypoints.push_back(kp);
        }
    }
    keypoints = filtered_keypoints;
}

void resizeImage(const Mat& input, Mat& output, double scale_factor) {
    // Check if the input image is empty
    if (input.empty()) {
        std::cerr << "Error: Input image is empty." << std::endl;
        return;
    }

    // Calculate new size based on the scale factor
    int new_width = cvRound(input.cols * scale_factor);
    int new_height = cvRound(input.rows * scale_factor);

    // Create output image with new size
    output.create(new_height, new_width, input.type());

    // Loop through each pixel in the output image
    for (int y = 0; y < new_height; ++y) {
        for (int x = 0; x < new_width; ++x) {
            // Map the coordinates in the output image to the coordinates in the input image
            float src_x = x / scale_factor;
            float src_y = y / scale_factor;

            // Get the four neighboring pixels in the input image
            int x1 = cvFloor(src_x);
            int y1 = cvFloor(src_y);
            int x2 = x1 + 1;
            int y2 = y1 + 1;

            // Ensure that the coordinates are within the input image boundaries
            x1 = std::max(0, std::min(x1, input.cols - 1));
            x2 = std::max(0, std::min(x2, input.cols - 1));
            y1 = std::max(0, std::min(y1, input.rows - 1));
            y2 = std::max(0, std::min(y2, input.rows - 1));

            // Calculate the interpolation weights
            float dx2 = src_x - x1;
            float dy2 = src_y - y1;
            float dx1 = 1.0 - dx2;
            float dy1 = 1.0 - dy2;

            // Perform bilinear interpolation
            output.at<Vec3b>(y, x) =
                input.at<Vec3b>(y1, x1) * dx1 * dy1 +
                input.at<Vec3b>(y1, x2) * dx2 * dy1 +
                input.at<Vec3b>(y2, x1) * dx1 * dy2 +
                input.at<Vec3b>(y2, x2) * dx2 * dy2;
        }
    }
}

// Function to build an image pyramid
void buildImagePyramid(const Mat& image, std::vector<Mat>& pyramid, int levels, double scale_factor) {
    // Check if the input image is empty
    if (image.empty()) {
        std::cerr << "Error: Input image is empty." << std::endl;
        return;
    }

    // Clear the pyramid vector
    pyramid.clear();

    // Add the original image to the pyramid
    pyramid.push_back(image.clone());

    // Resize and add subsequent levels to the pyramid
    Mat current_level = image.clone();
    for (int i = 1; i < levels; ++i) {
        // Resize the current level using the specified scale factor
        Size new_size(cvRound(current_level.cols * scale_factor), cvRound(current_level.rows * scale_factor));
        Mat next_level(new_size, current_level.type());
        resizeImage(current_level, next_level, scale_factor);

        // Add the resized image to the pyramid
        pyramid.push_back(next_level);

        // Update the current level for the next iteration
        current_level = next_level.clone();
    }
}

// Function to detect keypoints using difference of Gaussians
void detectKeypoints(const std::vector<Mat>& pyramid, std::vector<KeyPoint>& keypoints, double response_threshold) {
    Ptr<FeatureDetector> detector = ORB::create();
    for (int i = 0; i < pyramid.size(); ++i) {
        std::vector<KeyPoint> kps;
        detector->detect(pyramid[i], kps);
        // Filter keypoints based on response threshold
        filterKeypoints(kps, pyramid[i], response_threshold);
        keypoints.insert(keypoints.end(), kps.begin(), kps.end());
    }
}

void cartToPolarCustom(const Mat& grad_x, const Mat& grad_y, Mat& mag, Mat& angle, bool angleInDegrees) {
    // Ensure the input matrices have the same size
    if (grad_x.size() != grad_y.size()) {
        std::cerr << "Error: Gradient matrices must have the same size." << std::endl;
        return;
    }

    // Initialize output matrices
    mag.create(grad_x.size(), CV_32F);
    angle.create(grad_x.size(), CV_32F);

    // Loop through each pixel in the input matrices
    for (int y = 0; y < grad_x.rows; ++y) {
        for (int x = 0; x < grad_x.cols; ++x) {
            // Compute magnitude
            float dx = grad_x.at<float>(y, x);
            float dy = grad_y.at<float>(y, x);
            float magnitude = sqrt(dx * dx + dy * dy);
            mag.at<float>(y, x) = magnitude;

            // Compute angle
            float angle_radians = atan2(dy, dx);
            if (angleInDegrees) {
                // Convert angle to degrees
                angle.at<float>(y, x) = angle_radians * 180.0 / CV_PI;
            } else {
                angle.at<float>(y, x) = angle_radians;
            }
        }
    }
}
// Function to compute descriptors for keypoints
void computeDescriptors(const Mat& image, const std::vector<KeyPoint>& keypoints, Mat& descriptors) {
    // SIFT descriptor parameters
    const int descriptor_size = 128; // SIFT descriptor size
    const int patch_size = 16; // Size of the patch around each keypoint
    const int half_patch = patch_size / 2;

    // Compute gradients using Sobel operators
    Mat grad__x, grad__y;
    Sobel(image, grad__x, CV_32F, 1, 0);
    Sobel(image, grad__y, CV_32F, 0, 1);

    // Compute magnitude and orientation of gradients
    Mat mag, angle;
    cartToPolar(grad__x, grad__y, mag, angle, true);

    // Iterate over keypoints
    descriptors.create(keypoints.size(), descriptor_size, CV_32F);
    for (size_t i = 0; i < keypoints.size(); ++i) {
        // Get coordinates of keypoint
        int x = keypoints[i].pt.x;
        int y = keypoints[i].pt.y;

        // Extract patch around keypoint
        Mat patch;
        image(Rect(x - half_patch, y - half_patch, patch_size, patch_size)).copyTo(patch);

        // Compute histogram of gradients in the patch
        Mat hist(1, descriptor_size, CV_32F, Scalar(0));
        for (int r = 0; r < patch.rows; ++r) {
            for (int c = 0; c < patch.cols; ++c) {
                float patch_angle = angle.at<float>(y - half_patch + r, x - half_patch + c);
                int bin = int(patch_angle / 45); // 8 bins for 360 degrees
                hist.at<float>(0, bin) += mag.at<float>(y - half_patch + r, x - half_patch + c);
            }
        }

        // Normalize histogram
        normalize(hist, hist, 1, 0, NORM_L2);

        // Store histogram as descriptor
        hist.copyTo(descriptors.row(i));
    }
}

// Function to compute Hamming distance between two descriptors
int hammingDistance(const Mat& desc1, const Mat& desc2) {
    // Compute Hamming distance between two binary descriptors
    int distance = 0;
    for (int i = 0; i < desc1.cols; ++i) {
        distance += cv::norm(desc1.col(i) != desc2.col(i), NORM_L1);
    }
    return distance;
}

// Function to draw matches between two images
void drawMatchesFeatures(const Mat& image1, const std::vector<KeyPoint>& keypoints1,
    const Mat& image2, const std::vector<KeyPoint>& keypoints2,
    const std::vector<DMatch>& matches, Mat& output) {
    // Merge images horizontally
    int max_height = std::max(image1.rows, image2.rows);
    int total_width = image1.cols + image2.cols;
    output.create(max_height, total_width, CV_8UC3);
    output.setTo(Scalar(255, 255, 255)); // White background

    // Draw images
    Mat left_roi(output, Rect(0, 0, image1.cols, image1.rows));
    image1.copyTo(left_roi);
    Mat right_roi(output, Rect(image1.cols, 0, image2.cols, image2.rows));
    image2.copyTo(right_roi);

    // Draw matches with randomly chosen colors
    RNG rng; // Random number generator
    for (size_t i = 0; i < matches.size(); ++i) {
        const KeyPoint& kp1 = keypoints1[matches[i].queryIdx];
        const KeyPoint& kp2 = keypoints2[matches[i].trainIdx];

        // Offset keypoints for second image
        Point2f kp2_offset(kp2.pt.x + image1.cols, kp2.pt.y);

        // Generate random color
        Scalar color(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));

        // Draw line between matched keypoints with random color
        line(output, kp1.pt, kp2_offset, color, 1);

        // Draw circles around keypoints with random color
        circle(output, kp1.pt, 5, color, 2);
        circle(output, kp2_offset, 5, color, 2);
    }
}

// Function to compute Sum of Squared Differences (SSD) between two descriptors
// Function to compute Sum of Squared Differences (SSD) between two descriptors
double computeSSD(const Mat& desc1, const Mat& desc2) {
    // Ensure descriptor sizes are the same
    if (desc1.size() != desc2.size()) {
        std::cerr << "Error: Descriptor sizes are not the same." << std::endl;
        return -1;
    }

    // Compute SSD
    double ssd = 0.0;
    for (int i = 0; i < desc1.rows; ++i) {
        for (int j = 0; j < desc1.cols; ++j) {
            double diff = desc1.at<double>(i, j) - desc2.at<double>(i, j);
            ssd += diff * diff;
        }
    }
    return ssd;
}

// Function to compute Normalized Cross-Correlation (NCC) between two descriptors
double computeNCC(const Mat& desc1, const Mat& desc2) {
    // Ensure descriptor sizes are the same
    if (desc1.size() != desc2.size()) {
        std::cerr << "Error: Descriptor sizes are not the same." << std::endl;
        return -1;
    }

    // Compute mean and standard deviation of each descriptor
    double mean1 = 0.0, mean2 = 0.0;
    double sum_sq_diff1 = 0.0, sum_sq_diff2 = 0.0, sum_product = 0.0;
    for (int i = 0; i < desc1.rows; ++i) {
        for (int j = 0; j < desc1.cols; ++j) {
            mean1 += desc1.at<double>(i, j);
            mean2 += desc2.at<double>(i, j);
        }
    }
    mean1 /= (desc1.rows * desc1.cols);
    mean2 /= (desc2.rows * desc2.cols);

    // Compute correlation
    for (int i = 0; i < desc1.rows; ++i) {
        for (int j = 0; j < desc1.cols; ++j) {
            double diff1 = desc1.at<double>(i, j) - mean1;
            double diff2 = desc2.at<double>(i, j) - mean2;
            sum_sq_diff1 += diff1 * diff1;
            sum_sq_diff2 += diff2 * diff2;
            sum_product += diff1 * diff2;
        }
    }
    double stddev1 = sqrt(sum_sq_diff1 / (desc1.rows * desc1.cols));
    double stddev2 = sqrt(sum_sq_diff2 / (desc2.rows * desc2.cols));

    // Compute NCC
    double ncc = sum_product / (stddev1 * stddev2 * desc1.rows * desc1.cols);
    return ncc;
}

// Function to match descriptors between two sets using SSD or NCC
void matchDescriptors(const Mat& descriptors1, const Mat& descriptors2, std::vector<DMatch>& matches, std::string method) {
    // Clear existing matches
    matches.clear();

    // Loop through all descriptors in the first set
    for (int i = 0; i < descriptors1.rows; ++i) {
        int best_match_index = -1;
        double best_similarity = -std::numeric_limits<double>::max(); // Initialize with minimum value for SSD, maximum value for NCC

        // Compare the current descriptor with all descriptors in the second set
        for (int j = 0; j < descriptors2.rows; ++j) {
            // Compute similarity score using the specified method
            double similarity;
            if (method == "SSD") {
                similarity = computeSSD(descriptors1.row(i), descriptors2.row(j));
            }
            else if (method == "NCC") {
                similarity = computeNCC(descriptors1.row(i), descriptors2.row(j));
            }
            else {
                std::cerr << "Error: Invalid method specified." << std::endl;
                return;
            }

            // Update best match if a better similarity score is found
            if (similarity > best_similarity) {
                best_similarity = similarity;
                best_match_index = j;
            }
        }

        // Add the best match to the matches vector
        matches.push_back(DMatch(i, best_match_index, 0)); // Distance is not used for SSD and NCC
    }
}

// Main SIFT function
cv::Mat sift(const cv::Mat& image1, const cv::Mat& image2, SimilarityMetric metric, int levels, double scale_factor, double sigma, double response_threshold){
    // Parameters
    //int levels = 5; // Number of levels in the image pyramid
    //double scale_factor = 0.5; // Scaling factor for each level
    //double sigma = 1.6; // Standard deviation for Gaussian blur

    // Apply Gaussian blur to input images
    Mat blurred_image1 = applyGaussianBlur(image1, sigma);
    Mat blurred_image2 = applyGaussianBlur(image2, sigma);

    // Build image pyramids
    std::vector<Mat> pyramid1, pyramid2;
    buildImagePyramid(blurred_image1, pyramid1, levels, scale_factor);
    buildImagePyramid(blurred_image2, pyramid2, levels, scale_factor);

    // Detect keypoints
    std::vector<KeyPoint> keypoints1, keypoints2;
    detectKeypoints(pyramid1, keypoints1, 0.0);
    detectKeypoints(pyramid2, keypoints2, 0.0);

    // Compute descriptors
    Mat descriptors1, descriptors2;
    computeDescriptors(blurred_image1, keypoints1, descriptors1);
    computeDescriptors(blurred_image2, keypoints2, descriptors2);



    // Draw matches with SSD
    std::string metricName;
    switch (metric) {
    case SimilarityMetric::SSD:
        metricName = "SSD";
        break;
    case SimilarityMetric::NCC:
        metricName = "NCC";
        break;
    }



    Mat matched_image;
    if (metricName == "SSD") {
    std::vector<DMatch> matches_ssd;
    matchDescriptors(descriptors1, descriptors2, matches_ssd, metricName);
    drawMatchesFeatures(image1, keypoints1, image2, keypoints2, matches_ssd, matched_image);
    }

    else if(metricName == "NCC"){
        std::vector<DMatch> matches_ncc;
        matchDescriptors(descriptors1, descriptors2, matches_ncc, metricName);
        drawMatchesFeatures(image1, keypoints1, image2, keypoints2, matches_ncc, matched_image);
    }

    return matched_image;
}



void drawKeypoints(const Mat& image, const std::vector<KeyPoint>& keypoints, Mat& output) {
    // Draw circles around keypoints on the image with random colors
    output = image.clone();
    RNG rng; // Random number generator
    for (const auto& kp : keypoints) {
        Scalar color(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256)); // Random color
        circle(output, kp.pt, 5, color, 2); // Circle with radius 5
    }
}

Mat computeKeypoints(const Mat& image, std::vector<KeyPoint>& keypoints, Mat& descriptors,int levels ,double scale_factor , double sigma ,double response_threshold) {
    // Parameters
    //int levels = 4; // Number of levels in the image pyramid
    //double scale_factor = 0.5; // Scaling factor for each level
    //double sigma = 1.6; // Standard deviation for Gaussian blur
    // Draw keypoints on the image
    Mat image_with_keypoints;
    // Apply Gaussian blur to input image
    Mat blurred_image = applyGaussianBlur(image, sigma);

    // Build image pyramid
    std::vector<Mat> pyramid;
    buildImagePyramid(blurred_image, pyramid, levels, scale_factor);

    // Detect keypoints
    detectKeypoints(pyramid, keypoints, response_threshold);

    // Filter keypoints based on response value
    filterKeypoints(keypoints, blurred_image, response_threshold);

    // Compute descriptors
    computeDescriptors(blurred_image, keypoints, descriptors);
    if (response_threshold == 0.001) {
    drawKeypoints(image, keypoints, image_with_keypoints);
    return image_with_keypoints;
    }
    else {
    drawKeypoints(image, keypoints, image_with_keypoints);
    return image_with_keypoints;
    }
}

// int main() {
//     // Read input images
//     Mat image1 = imread("image1.jpg");
//     Mat image2 = imread("image3.jpg");
//     if (image1.empty() || image2.empty()) {
//         std::cerr << "Error: Unable to read input images." << std::endl;
//         return -1;
//     }

//     std::vector<KeyPoint> keypoints;
//     Mat descriptors;

//     computeKeypoints(image1, keypoints, descriptors, 0.001);
//     computeKeypoints(image1, keypoints, descriptors, 0.002);

//     // Timing for SIFT algorithm
//     auto start = high_resolution_clock::now(); // Start time
//     sift(image1, image2);
//     auto end = high_resolution_clock::now(); // End time
//     auto duration = duration_cast<milliseconds>(end - start); // Duration in milliseconds
//     long elapsedTime = duration.count(); // Elapsed time in milliseconds
//     long long elapsedTime_ns = elapsedTime * 1000000;
//     std::cout << "Elapsed time for SIFT algorithm: " << elapsedTime_ns << " ns." << std::endl;
//     std::cout << "Elapsed time for SIFT algorithm: " << elapsedTime << " milliseconds." << std::endl;

//     waitKey(0);
//     return 0;
// }
