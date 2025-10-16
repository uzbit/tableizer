#ifndef UTILITIES_HPP
#define UTILITIES_HPP

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "quad_analysis.hpp"

#define DEBUG_OUTPUT 0

#define DEBUG_POINT std::cout << "Reached " << __FILE__ << ":" << __LINE__ << std::endl;

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

#if DEBUG_OUTPUT
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
#else
#define LOGI(...)
#define LOGE(...)
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

vector<Point2f> orderQuad(const vector<Point2f>& pts);

struct TransformationResult {
    std::vector<cv::Point2f> transformedPoints;
    bool needsRotation;
    QuadOrientation orientation;
};

TransformationResult transformPointsUsingQuad(const std::vector<cv::Point2f>& points,
                                              const std::vector<cv::Point2f>& quadPoints,
                                              cv::Size imageSize, cv::Size displaySize,
                                              int inputRotationDegrees);

cv::Point2f perspectiveTransform(const cv::Point2f& point, const std::vector<cv::Point2f>& srcQuad,
                                 const std::vector<cv::Point2f>& dstRect);

cv::Point2f findUVInQuad(const cv::Point2f& point, const cv::Point2f& p0, const cv::Point2f& p1,
                         const cv::Point2f& p2, const cv::Point2f& p3);

static const std::string base64_chars =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789+/";

static inline bool isBase64(unsigned char c) { return (isalnum(c) || (c == '+') || (c == '/')); }

std::string base64Encode(unsigned char const* bytesToEncode, unsigned int inLen);
std::vector<unsigned char> base64Decode(std::string const& encodedString);

#endif  // UTILITIES_HPP
