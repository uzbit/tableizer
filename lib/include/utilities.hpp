#ifndef UTILITIES_HPP
#define UTILITIES_HPP

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

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

static const std::string base64_chars =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789+/";

static inline bool is_base64(unsigned char c) { return (isalnum(c) || (c == '+') || (c == '/')); }

std::string base64_encode(unsigned char const* bytes_to_encode, unsigned int in_len);
std::vector<unsigned char> base64_decode(std::string const& encoded_string);

#endif  // UTILITIES_HPP
