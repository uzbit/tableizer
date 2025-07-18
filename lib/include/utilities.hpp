#ifndef UTILITIES_HPP
#define UTILITIES_HPP

#include <opencv2/opencv.hpp>

#define DEBUG_POINT std::cout << "Reached " << __FILE__ << ":" << __LINE__ << std::endl;

using namespace std;
using namespace cv;

struct WarpResult {
    cv::Mat warped;     // The output image
    cv::Mat transform;  // 3×3 homography (float32)
};

WarpResult warpTable(const cv::Mat& bgrImg, const std::vector<cv::Point2f>& quad,
                     const std::string& imagePath, int outW = 1000, bool rotate = false,
                     double scaleF = 1.0);

// Function to order the quad points counter-clockwise
vector<Point2f> orderQuad(const vector<Point2f>& pts);

#endif  // UTILITIES_HPP