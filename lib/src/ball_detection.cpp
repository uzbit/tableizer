#include "ball_detection.hpp"
#include <opencv2/opencv.hpp>

std::string formatDetectionsJson(const std::vector<Detection>& detections) {
    std::string json = "{\"detections\": [";
    for (size_t i = 0; i < detections.size(); ++i) {
        const auto& detection = detections[i];
        json += "{";
        json += "\"class_id\": " + std::to_string(detection.classId) + ", ";
        json += "\"confidence\": " + std::to_string(detection.confidence) + ", ";
        json += "\"center_x\": " + std::to_string(detection.center.x) + ", ";
        json += "\"center_y\": " + std::to_string(detection.center.y) + ", ";
        json += "\"box\": {\"x\": " + std::to_string(detection.box.x) +
                ", \"y\": " + std::to_string(detection.box.y) +
                ", \"width\": " + std::to_string(detection.box.width) +
                ", \"height\": " + std::to_string(detection.box.height) + "}";
        json += "}";
        if (i < detections.size() - 1) {
            json += ", ";
        }
    }
    json += "]}";
    return json;
}

std::vector<Detection> parseDetectionsFromJson(const std::string& ballJsonStr) {
    std::vector<Detection> detections;
    size_t detectionsStart = ballJsonStr.find("\"detections\":");
    if (detectionsStart != std::string::npos) {
        size_t arrayStart = ballJsonStr.find("[", detectionsStart);
        size_t arrayEnd = ballJsonStr.find("]", arrayStart);
        if (arrayStart != std::string::npos && arrayEnd != std::string::npos) {
            size_t pos = arrayStart + 1;
            while (pos < arrayEnd) {
                size_t objStart = ballJsonStr.find("{", pos);
                if (objStart == std::string::npos || objStart >= arrayEnd) break;

                size_t objEnd = ballJsonStr.find("}", objStart);
                if (objEnd == std::string::npos) break;

                std::string detectionObj = ballJsonStr.substr(objStart, objEnd - objStart + 1);

                // Parse fields
                Detection det;
                size_t classIdPos = detectionObj.find("\"class_id\":");
                size_t confPos = detectionObj.find("\"confidence\":");
                size_t centerXPos = detectionObj.find("\"center_x\":");
                size_t centerYPos = detectionObj.find("\"center_y\":");
                size_t boxPos = detectionObj.find("\"box\":");

                if (classIdPos != std::string::npos) {
                    det.classId = std::stoi(detectionObj.substr(classIdPos + 11));
                }
                if (confPos != std::string::npos) {
                    det.confidence = std::stof(detectionObj.substr(confPos + 14));
                }
                if (centerXPos != std::string::npos) {
                    det.center.x = std::stof(detectionObj.substr(centerXPos + 12));
                }
                if (centerYPos != std::string::npos) {
                    det.center.y = std::stof(detectionObj.substr(centerYPos + 12));
                }
                if (boxPos != std::string::npos) {
                    size_t xPos = detectionObj.find("\"x\":", boxPos);
                    size_t yPos = detectionObj.find("\"y\":", boxPos);
                    size_t wPos = detectionObj.find("\"width\":", boxPos);
                    size_t hPos = detectionObj.find("\"height\":", boxPos);
                    if (xPos != std::string::npos && yPos != std::string::npos &&
                        wPos != std::string::npos && hPos != std::string::npos) {
                        int x = std::stoi(detectionObj.substr(xPos + 4));
                        int y = std::stoi(detectionObj.substr(yPos + 4));
                        int w = std::stoi(detectionObj.substr(wPos + 8));
                        int h = std::stoi(detectionObj.substr(hPos + 9));
                        det.box = cv::Rect(x, y, w, h);
                    }
                }

                detections.push_back(det);
                pos = objEnd + 1;
            }
        }
    }
    return detections;
}
