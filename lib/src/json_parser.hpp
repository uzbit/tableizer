#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

struct JsonParseResult {
    std::vector<cv::Point2f> quadPoints;
    cv::Mat mask;
    bool success;
};

JsonParseResult parseTableDetectionJson(const std::string& jsonStr);
std::vector<cv::Point2f> parseQuadPointsFromJson(const std::string& jsonStr);
cv::Mat parseMaskFromJson(const std::string& jsonStr);
