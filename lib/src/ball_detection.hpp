#pragma once

#include <string>
#include <vector>
#include "ball_detector.hpp"

std::string formatDetectionsJson(const std::vector<Detection>& detections);
std::vector<Detection> parseDetectionsFromJson(const std::string& jsonStr);
