#include "json_parser.hpp"
#include "base64_utils.hpp"
#include "utilities.hpp"

std::vector<cv::Point2f> parseQuadPointsFromJson(const std::string& jsonStr) {
    std::vector<cv::Point2f> quadPoints;
    size_t quadStart = jsonStr.find("\"quad_points\":");
    if (quadStart == std::string::npos) {
        return quadPoints;
    }

    size_t arrayStart = jsonStr.find("[", quadStart);
    if (arrayStart != std::string::npos) {
        size_t pos = arrayStart + 1;
        int pointCount = 0;

        while (pos < jsonStr.length() && pointCount < 4) {
            size_t pointStart = jsonStr.find("[", pos);
            if (pointStart == std::string::npos) break;

            size_t pointEnd = jsonStr.find("]", pointStart);
            if (pointEnd == std::string::npos) break;

            std::string pointStr = jsonStr.substr(pointStart + 1, pointEnd - pointStart - 1);
            size_t comma = pointStr.find(",");
            if (comma != std::string::npos) {
                float xCoord = std::stof(pointStr.substr(0, comma));
                float yCoord = std::stof(pointStr.substr(comma + 1));
                quadPoints.emplace_back(xCoord, yCoord);
                pointCount++;
            }

            pos = pointEnd + 1;
        }
    }
    return quadPoints;
}

cv::Mat parseMaskFromJson(const std::string& jsonStr) {
    cv::Mat mask;
    size_t maskStart = jsonStr.find("\"mask\":");
    if (maskStart != std::string::npos) {
        size_t dataStart = jsonStr.find("\"data\":", maskStart);
        if (dataStart != std::string::npos) {
            size_t dataValueStart = jsonStr.find("\"", dataStart + 7) + 1;
            size_t dataValueEnd = jsonStr.find("\"", dataValueStart);
            if (dataValueEnd != std::string::npos) {
                std::string maskBase64 =
                    jsonStr.substr(dataValueStart, dataValueEnd - dataValueStart);

                LOGI("Found mask data (%zu bytes base64)", maskBase64.length());
                try {
                    std::vector<unsigned char> maskData = Base64Utils::decode(maskBase64);
                    if (!maskData.empty()) {
                        mask = cv::imdecode(maskData, cv::IMREAD_GRAYSCALE);
                        if (!mask.empty()) {
                            LOGI("Successfully decoded mask: %dx%d", mask.cols, mask.rows);
                        } else {
                            LOGE("Failed to decode mask PNG data");
                        }
                    }
                } catch (const std::exception& e) {
                    LOGE("Error decoding mask: %s", e.what());
                }
            }
        }
    }
    return mask;
}

JsonParseResult parseTableDetectionJson(const std::string& jsonStr) {
    JsonParseResult result;
    result.quadPoints = parseQuadPointsFromJson(jsonStr);
    result.mask = parseMaskFromJson(jsonStr);
    result.success = !result.quadPoints.empty();
    return result;
}
