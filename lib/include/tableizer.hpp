#ifndef TABLEIZER_HPP
#define TABLEIZER_HPP

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "ball_detector.hpp"

using namespace cv;

int runTableizerForImage(Mat image, BallDetector& ballDetector);

cv::Mat createMaskedImage(const cv::Mat& image, const std::vector<cv::Point2f>& quadPoints);

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default"))) __attribute__((used)) void* initialize_detector(
    const char* modelPath);

__attribute__((visibility("default"))) __attribute__((used)) const char* detect_balls_bgra(
    void* detectorPtr, const unsigned char* imageBytes, int width, int height, int stride,
    const float* quadPoints, int quadPointsLength, int channelFormat);

__attribute__((visibility("default"))) __attribute__((used)) const char* detect_table_bgra(
    const unsigned char* imageBytes, int width, int height, int stride, int channelFormat);

__attribute__((visibility("default"))) __attribute__((used)) void release_detector(
    void* detectorPtr);

__attribute__((visibility("default"))) __attribute__((used)) const char*
transform_points_using_quad(const float* pointsData, int pointsCount, const float* quadData,
                            int quadCount, int imageWidth, int imageHeight, int displayWidth,
                            int displayHeight, int inputRotationDegrees);

__attribute__((visibility("default"))) __attribute__((used)) const char* normalize_image_bgra(
    const unsigned char* inputBytes, int inputWidth, int inputHeight, int inputStride,
    int rotationDegrees, unsigned char* outputBytes, int outputBufferSize, int channelFormat);

#ifdef __cplusplus
}
#endif

#endif  // TABLEIZER_HPP
