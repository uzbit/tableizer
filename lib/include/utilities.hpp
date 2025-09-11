#ifndef UTILITIES_HPP
#define UTILITIES_HPP

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#define DEBUG_POINT std::cout << "Reached " << __FILE__ << ":" << __LINE__ << std::endl;

// Platform detection macros
#if defined(__ANDROID__)
#define PLATFORM_ANDROID 1
#elif defined(__APPLE__)
#include <TargetConditionals.h>
#if TARGET_OS_IPHONE
#define PLATFORM_IOS 1
#else
#define PLATFORM_MACOS 1
#endif
#elif defined(_WIN32) || defined(_WIN64)
#define PLATFORM_WINDOWS 1
#else
#define PLATFORM_LINUX 1
#endif

// Conditional logging headers and macros
#ifdef PLATFORM_ANDROID
#include <android/log.h>
#define LOG_TAG "tableizer_native"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#elif defined(PLATFORM_IOS)
#include <os/log.h>
#define LOGI(...) os_log(OS_LOG_DEFAULT, __VA_ARGS__)
#define LOGE(...) os_log_error(OS_LOG_DEFAULT, __VA_ARGS__)
#else
#include <cstdio>
#define LOGI(...)            \
    do {                     \
        printf("INFO: ");    \
        printf(__VA_ARGS__); \
        printf("\n");        \
    } while (0)
#define LOGE(...)                     \
    do {                              \
        fprintf(stderr, "ERROR: ");   \
        fprintf(stderr, __VA_ARGS__); \
        fprintf(stderr, "\n");        \
    } while (0)
#endif

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

// Coordinate transformation functions
struct TransformationResult {
    std::vector<cv::Point2f> transformedPoints;
    bool needsRotation;
};

// Transform points using quad-to-rectangle perspective transformation
TransformationResult transformPointsUsingQuad(
    const std::vector<cv::Point2f>& points,
    const std::vector<cv::Point2f>& quadPoints,
    cv::Size imageSize,
    cv::Size displaySize
);

// Helper functions for perspective transformation
cv::Point2f perspectiveTransform(
    const cv::Point2f& point,
    const std::vector<cv::Point2f>& srcQuad,
    const std::vector<cv::Point2f>& dstRect
);

cv::Point2f findUVInQuad(
    const cv::Point2f& point,
    const cv::Point2f& p0, const cv::Point2f& p1,
    const cv::Point2f& p2, const cv::Point2f& p3
);

static const std::string base64_chars =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789+/";

static inline bool is_base64(unsigned char c) { return (isalnum(c) || (c == '+') || (c == '/')); }

std::string base64_encode(unsigned char const* bytes_to_encode, unsigned int in_len);
std::vector<unsigned char> base64_decode(std::string const& encoded_string);

#endif  // UTILITIES_HPP
