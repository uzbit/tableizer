#ifndef UTILITIES_HPP
#define UTILITIES_HPP

#include <opencv2/opencv.hpp>

#define DEBUG_POINT std::cout << "Reached " << __FILE__ << ":" << __LINE__ << std::endl;

struct WarpResult {
    cv::Mat warped;     // The output image
    cv::Mat transform;  // 3×3 homography (float32)
};

WarpResult warpTable(const cv::Mat& bgrImg, const std::vector<cv::Point2f>& quad,
                     const std::string& imagePath, int outW = 1000, bool rotate = false,
                     double scaleF = 1.0);

#endif  // UTILITIES_HPP