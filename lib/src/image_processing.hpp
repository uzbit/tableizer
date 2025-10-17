#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include "ball_detector.hpp"

void drawBallsOnImages(const std::vector<Detection>& detections, cv::Mat& warpedOut,
                       cv::Mat& shotStudio, const cv::Mat& transform);

cv::Mat createMaskedImage(const cv::Mat& image, const std::vector<cv::Point2f>& quadPoints);
