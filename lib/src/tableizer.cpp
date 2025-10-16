#include "tableizer.hpp"

#include <algorithm>
#include <iostream>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <vector>

#include "ball_detector.hpp"
#include "base64_utils.hpp"
#include "quad_analysis.hpp"
#include "table_detector.hpp"
#include "utilities.hpp"

using namespace std;
using namespace cv;

struct JsonParseResult {
    std::vector<cv::Point2f> quadPoints;
    cv::Mat mask;
    bool success;
};

JsonParseResult parseTableDetectionJson(const std::string& jsonStr);
std::vector<cv::Point2f> parseQuadPointsFromJson(const std::string& jsonStr);
cv::Mat parseMaskFromJson(const std::string& jsonStr);
void drawBallsOnImages(const std::vector<Detection>& detections, cv::Mat& warpedOut,
                       cv::Mat& shotStudio, const cv::Mat& transform);

#define CONF_THRESH 0.5
#define IOU_THRESH 0.5

// Table detection constants
#define CELL_SIZE 10
#define DELTAE_THRESH 15.0
#define RESIZE_MAX_SIZE 800

int runTableizerForImage(Mat image, BallDetector& ballDetector) {
#ifdef PLATFORM_MACOS
    LOGI("--- 1: Table Detection (FFI) ---");
    cv::Mat bgraImage;
    cv::cvtColor(image, bgraImage, cv::COLOR_BGR2BGRA);
    const char* jsonResult =
        detect_table_bgra(bgraImage.data, bgraImage.cols, bgraImage.rows, bgraImage.step);

    if (jsonResult == nullptr) {
        LOGE("Error: FFI detection returned null.");
        return -1;
    }

    std::string jsonStr(jsonResult);
    JsonParseResult parseResult = parseTableDetectionJson(jsonStr);

    if (!parseResult.success) {
        LOGE("Error: No quad_points found in JSON response.");
        return -1;
    }

    std::vector<cv::Point2f> quadPoints = parseResult.quadPoints;
    cv::Mat quadMask = parseResult.mask;

    float topLength = cv::norm(quadPoints[1] - quadPoints[0]);    // top edge
    float rightLength = cv::norm(quadPoints[2] - quadPoints[1]);  // right edge
    bool rotate = topLength < rightLength * 1.75;
    LOGI("Quad analysis: topLength=%.2f, rightLength=%.2f, rotate=%s", topLength, rightLength,
         rotate ? "true" : "false");
    WarpResult warpResult = warpTable(image, quadPoints, "warp.jpg", 840, rotate);

    LOGI("--- Step 4: Ball Detection (FFI) ---");

    // Call detect_balls_bgra FFI
    void* detectorPtr = static_cast<void*>(&ballDetector);

    // Convert quadPoints to a flat float array for the FFI call
    std::vector<float> quadPointsFlat;
    for (const auto& p : quadPoints) {
        quadPointsFlat.push_back(p.x);
        quadPointsFlat.push_back(p.y);
    }

    const char* ballJsonResult =
        detect_balls_bgra(detectorPtr, bgraImage.data, bgraImage.cols, bgraImage.rows,
                          bgraImage.step, quadPointsFlat.data(), quadPoints.size());

    if (ballJsonResult == nullptr) {
        LOGE("Error: Ball detection FFI returned null.");
        return -1;
    }

    // Parse ball detection results
    std::string ballJsonStr(ballJsonResult);
    LOGI("Ball detection result: %s", ballJsonStr.c_str());

    // Parse detections from JSON
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

    LOGI("Parsed %zu ball detections", detections.size());

    // Visualize results
    cv::Mat Hwarp;
    warpResult.transform.convertTo(Hwarp, CV_64F);
    cv::Mat Htotal = Hwarp;

    cv::Mat shotStudio;
    string studioPath =
        "/Users/uzbit/Documents/projects/tableizer/data/shotstudio_table_felt_only.png";
    shotStudio = cv::imread(studioPath);
    cv::Mat warpedOut = warpResult.warped.clone();

    drawBallsOnImages(detections, warpedOut, shotStudio, Htotal);

    LOGI("Image size before imshow (Warped Table + Balls): %dx%d", warpedOut.cols, warpedOut.rows);
    cv::imshow("Warped Table + Balls", warpedOut);
    if (!shotStudio.empty()) {
        LOGI("Image size before imshow (Shot-Studio Overlay): %dx%d", shotStudio.cols,
             shotStudio.rows);
        cv::imshow("Shot-Studio Overlay", shotStudio);
    }
    cv::waitKey(0);
#endif

    return 0;
}

string formatDetectionsJson(const vector<Detection>& detections) {
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

void drawBallsOnImages(const std::vector<Detection>& detections, cv::Mat& warpedOut,
                       cv::Mat& shotStudio, const cv::Mat& transform) {
    if (detections.empty()) {
        LOGI("No balls detected.");
        return;
    }

    vector<cv::Point2f> ballCentersOrig;
    for (const auto& d : detections) {
        ballCentersOrig.emplace_back(d.center);
    }

    vector<cv::Point2f> ballCentersCanonical;
    cv::perspectiveTransform(ballCentersOrig, ballCentersCanonical, transform);

    int tableSizeInches = 100;
    int longEdgePx = std::max(warpedOut.cols, warpedOut.rows);
    int radius = std::max((int)round(longEdgePx * (2.25 / tableSizeInches) / 2.0), 4);
    float textSize = 0.7 * (radius / 8.0);
    const cv::Scalar textColor(255, 255, 255);

    for (size_t i = 0; i < ballCentersCanonical.size(); ++i) {
        const auto& ballPosition = ballCentersCanonical[i];
        LOGI("  • class %d conf %.3f @ (%.1f, %.1f)", detections[i].classId,
             detections[i].confidence, ballPosition.x, ballPosition.y);

        cv::Scalar ballColor;
        switch (detections[i].classId) {
            case 3:
                ballColor = cv::Scalar(0, 0, 255);
                break;
            case 2:
                ballColor = cv::Scalar(255, 222, 33);
                break;
            case 1:
                ballColor = cv::Scalar(255, 255, 255);
                break;
            case 0:
                ballColor = cv::Scalar(0, 0, 0);
                break;
            default:
                ballColor = cv::Scalar(128, 128, 128);
                break;
        }

        cv::circle(warpedOut, ballPosition, radius, ballColor, cv::FILLED, cv::LINE_AA);
        cv::putText(warpedOut, std::to_string(detections[i].classId),
                    ballPosition + cv::Point2f(radius + 2, 0), cv::FONT_HERSHEY_SIMPLEX, textSize,
                    textColor, 2);

        if (!shotStudio.empty()) {
            cv::circle(shotStudio, ballPosition, radius, ballColor, cv::FILLED);
            cv::putText(shotStudio, std::to_string(detections[i].classId),
                        ballPosition + cv::Point2f(radius + 2, 0), cv::FONT_HERSHEY_SIMPLEX,
                        textSize, textColor, 2);
        }
    }
}

cv::Mat createMaskedImage(const cv::Mat& image, const std::vector<cv::Point2f>& quadPoints) {
    cv::Mat mask = cv::Mat::zeros(image.size(), CV_8UC1);
    cv::Mat maskedImage;

    if (quadPoints.size() == 4) {
        std::vector<cv::Point> quadPointsInt;
        for (const auto& pt : quadPoints) {
            quadPointsInt.emplace_back(cv::Point(cvRound(pt.x), cvRound(pt.y)));
        }
        cv::fillConvexPoly(mask, quadPointsInt, cv::Scalar(255));
        image.copyTo(maskedImage, mask);

        // Draw the quad edges with different colors
        // Edge definitions: 0=TL, 1=TR, 2=BR, 3=BL
        cv::line(maskedImage, quadPoints[0], quadPoints[1], cv::Scalar(0, 0, 255), 2);  // Top: Red
        cv::line(maskedImage, quadPoints[1], quadPoints[2], cv::Scalar(0, 255, 0),
                 2);  // Right: Green
        cv::line(maskedImage, quadPoints[2], quadPoints[3], cv::Scalar(255, 0, 0),
                 2);  // Bottom: Blue
        cv::line(maskedImage, quadPoints[3], quadPoints[0], cv::Scalar(0, 255, 255),
                 2);  // Left: Yellow
    } else {
        // If no quad, return a copy of the original image
        image.copyTo(maskedImage);
    }

    return maskedImage;
}

extern "C" {

void* initialize_detector(const char* modelPath) {
    try {
        LOGI("initialize_detector called with path: %s", modelPath ? modelPath : "NULL");
        BallDetector* detector = new BallDetector(modelPath);
        LOGI("BallDetector created successfully");
        return static_cast<void*>(detector);
    } catch (const std::exception& e) {
        LOGE("Error initializing detector: %s", e.what());
        return nullptr;
    } catch (...) {
        LOGE("Unknown error initializing detector");
        return nullptr;
    }
}

const char* detect_balls_bgra(void* detectorPtr, const unsigned char* imageBytes, int width,
                              int height, int stride, const float* quadPoints,
                              int quadPointsLength) {
    static std::string resultStr;
    if (!detectorPtr) {
        resultStr = "{\"error\": \"Invalid detector instance\"}";
        return resultStr.c_str();
    }
    try {
        LOGI("[detect_balls_bgra] Input: %dx%d, stride: %d", width, height, stride);
        LOGI(
            "[detect_balls_bgra] First 16 bytes: %d,%d,%d,%d | %d,%d,%d,%d | %d,%d,%d,%d | "
            "%d,%d,%d,%d",
            imageBytes[0], imageBytes[1], imageBytes[2], imageBytes[3], imageBytes[4],
            imageBytes[5], imageBytes[6], imageBytes[7], imageBytes[8], imageBytes[9],
            imageBytes[10], imageBytes[11], imageBytes[12], imageBytes[13], imageBytes[14],
            imageBytes[15]);

        cv::Mat bgraImage(height, width, CV_8UC4, (void*)imageBytes, stride);
        if (bgraImage.empty()) {
            LOGE("[detect_balls_bgra] Failed to create Mat from bytes");
            resultStr = "{\"error\": \"Failed to create image from bytes\"}";
            return resultStr.c_str();
        }

        LOGI("[detect_balls_bgra] Created Mat: %dx%d, channels: %d", bgraImage.cols, bgraImage.rows,
             bgraImage.channels());

        // Sample a pixel from the center to verify color values
        cv::Vec4b centerPixel = bgraImage.at<cv::Vec4b>(height / 2, width / 2);
        LOGI("[detect_balls_bgra] Center pixel (BGRA): B=%d G=%d R=%d A=%d", centerPixel[0],
             centerPixel[1], centerPixel[2], centerPixel[3]);

        cv::Mat imageRgb;
        // Convert from BGRA to RGB (YOLO expects RGB format)
        cv::cvtColor(bgraImage, imageRgb, cv::COLOR_BGRA2RGB);

        LOGI("[detect_balls_bgra] Converted to RGB: %dx%d, channels: %d", imageRgb.cols,
             imageRgb.rows, imageRgb.channels());

        // Sample the same pixel after conversion
        cv::Vec3b centerPixelRgb = imageRgb.at<cv::Vec3b>(height / 2, width / 2);
        LOGI("[detect_balls_bgra] Center pixel (RGB): R=%d G=%d B=%d", centerPixelRgb[0],
             centerPixelRgb[1], centerPixelRgb[2]);

        // Apply mask if provided
        cv::Mat imageForDetection;
        if (quadPoints != nullptr && quadPointsLength == 4) {
            std::vector<cv::Point2f> quadPointsVec;
            for (int i = 0; i < quadPointsLength; ++i) {
                quadPointsVec.emplace_back(quadPoints[i * 2], quadPoints[i * 2 + 1]);
            }
            LOGI("[detect_balls_bgra] Applying mask from %d quad points", quadPointsLength);
            imageForDetection = createMaskedImage(imageRgb, quadPointsVec);
            LOGI("[detect_balls_bgra] Mask applied successfully");

#if defined(PLATFORM_MACOS) || defined(PLATFORM_LINUX) || defined(PLATFORM_WINDOWS)
            LOGI("Image size before imshow (Masked Image for Ball Detection): %dx%d",
                 imageForDetection.cols, imageForDetection.rows);
            cv::imshow("Masked Image for Ball Detection", imageForDetection);
            cv::waitKey(0);  // Wait for user to inspect both windows
#endif
        } else {
            LOGI("[detect_balls_bgra] No mask provided, using full image");
            imageForDetection = imageRgb;
        }

#if DEBUG_OUTPUT
        // Write debug image showing what's being sent to ball detection
        cv::imwrite("/sdcard/Download/ball_debug.jpg", imageForDetection);
        LOGI("[detect_balls_bgra] Wrote debug image to: /sdcard/Download/ball_debug.jpg");
#endif
        const auto detections = static_cast<BallDetector*>(detectorPtr)
                                    ->detect(imageForDetection, CONF_THRESH, IOU_THRESH);

        LOGI("[detect_balls_bgra] Detection complete: %zu objects found", detections.size());

        resultStr = formatDetectionsJson(detections);
        return resultStr.c_str();
    } catch (const std::exception& e) {
        LOGE("[detect_balls_bgra] Exception: %s", e.what());
        resultStr = std::string("{\"error\": \"") + e.what() + "\"}";
        return resultStr.c_str();
    }
}

void release_detector(void* detectorPtr) {
    if (detectorPtr) {
        delete static_cast<BallDetector*>(detectorPtr);
    }
}

const char* detect_table_bgra(const unsigned char* imageBytes, int width, int height, int stride) {
    static std::string resultStr;
    try {
        cv::Mat bgraImageUnrotated(height, width, CV_8UC4, (void*)imageBytes, stride);
        if (bgraImageUnrotated.empty()) {
            LOGE("Failed to create image from bytes.");
            resultStr = "{\"error\": \"Failed to create image from bytes\"}";
            return resultStr.c_str();
        }

        TableDetector tableDetector(RESIZE_MAX_SIZE, CELL_SIZE, DELTAE_THRESH);
        Mat cellularMask, tableDetection;
        tableDetector.detect(bgraImageUnrotated, cellularMask, tableDetection, 0);

        std::vector<cv::Point2f> quadPoints = tableDetector.getQuadFromMask(cellularMask);

        float scaleX = (float)bgraImageUnrotated.cols / tableDetection.cols;
        float scaleY = (float)bgraImageUnrotated.rows / tableDetection.rows;

        std::vector<cv::Point2f> scaledQuadPoints;
        for (const auto& pt : quadPoints) {
            scaledQuadPoints.emplace_back(pt.x * scaleX, pt.y * scaleY);
        }

        // Analyze quad orientation
        QuadOrientation orientation = OTHER;
        std::string orientationStr = "OTHER";
        ViewValidation validation;

        if (scaledQuadPoints.size() == 4) {
            orientation = QuadAnalysis::orientation(scaledQuadPoints);
            orientationStr = QuadAnalysis::orientationToString(orientation);
            LOGI("[detect_table_bgra] Table orientation: %s", orientationStr.c_str());

            // Log image context for debugging coordinate system
            LOGI("[detect_table_bgra] === IMAGE CONTEXT ===");
            LOGI("[detect_table_bgra] Image dimensions: %dx%d (WxH), aspect=%.3f",
                 bgraImageUnrotated.cols, bgraImageUnrotated.rows,
                 (float)bgraImageUnrotated.cols / bgraImageUnrotated.rows);

            // Log quad points as percentage of image dimensions
            LOGI("[detect_table_bgra] Quad points as %% of image:");
            for (int i = 0; i < 4; i++) {
                float xPct = (scaledQuadPoints[i].x / bgraImageUnrotated.cols) * 100.0f;
                float yPct = (scaledQuadPoints[i].y / bgraImageUnrotated.rows) * 100.0f;
                LOGI("  [%d] %.1f%% width, %.1f%% height", i, xPct, yPct);
            }

            // Validate landscape 16:9 short-side view
            validation = QuadAnalysis::validateLandscapeShortSideView(
                scaledQuadPoints, cv::Size(bgraImageUnrotated.cols, bgraImageUnrotated.rows),
                orientation);
        } else {
            // No valid quad - set validation to invalid
            validation.isValid = false;
            validation.isLandscape = false;
            validation.isCorrectAspectRatio = false;
            validation.isShortSideView = false;
            validation.imageAspectRatio =
                (float)bgraImageUnrotated.cols / (float)bgraImageUnrotated.rows;
            validation.errorMessage = "No valid quad detected";
        }

        std::string json = "{\"quad_points\": [";
        for (size_t i = 0; i < scaledQuadPoints.size(); ++i) {
            json += "[" + std::to_string(scaledQuadPoints[i].x) + ", " +
                    std::to_string(scaledQuadPoints[i].y) + "]";
            if (i < scaledQuadPoints.size() - 1) json += ", ";
        }
        json += "], \"image_width\": " + std::to_string(bgraImageUnrotated.cols) +
                ", \"image_height\": " + std::to_string(bgraImageUnrotated.rows);

        json += ", \"orientation\": \"" + orientationStr + "\"";

        // Add validation results
        json += ", \"validation\": {";
        json += "\"is_valid\": " + std::string(validation.isValid ? "true" : "false") + ", ";
        json +=
            "\"is_landscape\": " + std::string(validation.isLandscape ? "true" : "false") + ", ";
        json += "\"is_correct_aspect_ratio\": " +
                std::string(validation.isCorrectAspectRatio ? "true" : "false") + ", ";
        json += "\"is_short_side_view\": " +
                std::string(validation.isShortSideView ? "true" : "false") + ", ";
        json += "\"image_aspect_ratio\": " + std::to_string(validation.imageAspectRatio);
        if (!validation.errorMessage.empty()) {
            json += ", \"error_message\": \"" + validation.errorMessage + "\"";
        }
        json += "}";

        json += "}";

#if DEBUG_OUTPUT

        // Write debug image showing detected quad
        if (scaledQuadPoints.size() == 4) {
            cv::Mat debugImage;
            cv::cvtColor(bgraImageUnrotated, debugImage, cv::COLOR_BGRA2BGR);

            // Draw the quad with colored edges
            // Edge definitions: 0=TL, 1=TR, 2=BR, 3=BL
            cv::line(debugImage, scaledQuadPoints[0], scaledQuadPoints[1], cv::Scalar(0, 0, 255),
                     3);  // Top: Red
            cv::line(debugImage, scaledQuadPoints[1], scaledQuadPoints[2], cv::Scalar(0, 255, 0),
                     3);  // Right: Green
            cv::line(debugImage, scaledQuadPoints[2], scaledQuadPoints[3], cv::Scalar(255, 0, 0),
                     3);  // Bottom: Blue
            cv::line(debugImage, scaledQuadPoints[3], scaledQuadPoints[0], cv::Scalar(0, 255, 255),
                     3);  // Left: Yellow

            // Draw corner points with numbers
            for (int i = 0; i < 4; i++) {
                cv::circle(debugImage, scaledQuadPoints[i], 10, cv::Scalar(255, 255, 255), -1);
                cv::putText(debugImage, std::to_string(i), scaledQuadPoints[i] + cv::Point2f(-5, 5),
                            cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 0), 2);
            }

            cv::imwrite("/sdcard/Download/table_debug.jpg", debugImage);
            LOGI("[detect_table_bgra] Wrote debug image to /sdcard/Download/table_debug.jpg");
        }
#endif

        resultStr = json;
        return resultStr.c_str();

    } catch (const std::exception& e) {
        LOGE("Error in detect_table_bgra: %s", e.what());
        resultStr =
            "{\"error\": \"Exception in detect_table_bgra: " + std::string(e.what()) + "\"}";
        return resultStr.c_str();
    }
}

const char* transform_points_using_quad(const float* pointsData, int pointsCount,
                                        const float* quadData, int quadCount, int imageWidth,
                                        int imageHeight, int displayWidth, int displayHeight,
                                        int inputRotationDegrees) {
    static std::string resultStr;

    try {
        if (pointsCount == 0 || quadCount != 4) {
            resultStr = "{\"error\": \"Invalid input parameters\"}";
            return resultStr.c_str();
        }

        std::vector<cv::Point2f> points;
        for (int i = 0; i < pointsCount; i++) {
            points.emplace_back(pointsData[i * 2], pointsData[i * 2 + 1]);
        }

        std::vector<cv::Point2f> quadPoints;
        for (int i = 0; i < quadCount; i++) {
            quadPoints.emplace_back(quadData[i * 2], quadData[i * 2 + 1]);
        }

        // Log input parameters received from Dart
        LOGI("[transform_points] === C++ TRANSFORM INPUT ===");
        LOGI("[transform_points] Received %d ball points, %d quad points", pointsCount, quadCount);
        LOGI("[transform_points] Image size: %dx%d, Display size: %dx%d", imageWidth, imageHeight,
             displayWidth, displayHeight);
        LOGI("[transform_points] Input rotation: %d degrees", inputRotationDegrees);
        LOGI("[transform_points] Quad points (in image %dx%d space):", imageWidth, imageHeight);
        for (int i = 0; i < quadCount; i++) {
            LOGI("  Quad[%d]: (%.1f, %.1f)", i, quadPoints[i].x, quadPoints[i].y);
        }
        LOGI("[transform_points] Ball positions (in image %dx%d space, before transform):",
             imageWidth, imageHeight);
        int ballsToLog = std::min(5, pointsCount);
        for (int i = 0; i < ballsToLog; i++) {
            LOGI("  Ball[%d]: (%.1f, %.1f)", i, points[i].x, points[i].y);
        }
        if (pointsCount > 5) {
            LOGI("  ... and %d more balls", pointsCount - 5);
        }

        TransformationResult transformation =
            transformPointsUsingQuad(points, quadPoints, cv::Size(imageWidth, imageHeight),
                                     cv::Size(displayWidth, displayHeight), inputRotationDegrees);

        // Log transformed output
        LOGI("[transform_points] === C++ TRANSFORM OUTPUT ===");
        LOGI("[transform_points] Orientation detected: %s",
             QuadAnalysis::orientationToString(transformation.orientation).c_str());
        LOGI("[transform_points] Needs rotation: %s",
             transformation.needsRotation ? "true" : "false");
        LOGI("[transform_points] Transformed ball positions (in display %dx%d space):",
             displayWidth, displayHeight);
        int transformedToLog = std::min(5, (int)transformation.transformedPoints.size());
        for (int i = 0; i < transformedToLog; i++) {
            LOGI("  Ball[%d]: (%.1f, %.1f)", i, transformation.transformedPoints[i].x,
                 transformation.transformedPoints[i].y);
        }
        if ((int)transformation.transformedPoints.size() > 5) {
            LOGI("  ... and %d more balls", (int)transformation.transformedPoints.size() - 5);
        }
        LOGI("[transform_points] === END TRANSFORM ===");

        std::string json = "{";
        json += "\"transformed_points\": [";
        for (size_t i = 0; i < transformation.transformedPoints.size(); ++i) {
            const auto& pt = transformation.transformedPoints[i];
            json += "{\"x\": " + std::to_string(pt.x) + ", \"y\": " + std::to_string(pt.y) + "}";
            if (i < transformation.transformedPoints.size() - 1) json += ", ";
        }
        json += "], ";
        json +=
            "\"needs_rotation\": " + std::string(transformation.needsRotation ? "true" : "false");
        json += ", \"orientation\": \"" +
                QuadAnalysis::orientationToString(transformation.orientation) + "\"";
        json += "}";

        resultStr = json;
        return resultStr.c_str();

    } catch (const std::exception& e) {
        resultStr = "{\"error\": \"" + std::string(e.what()) + "\"}";
        return resultStr.c_str();
    }
}

const char* normalize_image_bgra(const unsigned char* inputBytes, int inputWidth, int inputHeight,
                                 int inputStride, int rotationDegrees, unsigned char* outputBytes,
                                 int outputBufferSize) {
    static std::string resultStr;

    try {
        LOGI("[normalize_image_bgra] Input: %dx%d, stride: %d, rotation: %d", inputWidth,
             inputHeight, inputStride, rotationDegrees);

        // Create Mat from input bytes
        cv::Mat inputImage(inputHeight, inputWidth, CV_8UC4, (void*)inputBytes, inputStride);
        if (inputImage.empty()) {
            LOGE("[normalize_image_bgra] Failed to create Mat from input bytes");
            resultStr = "{\"error\": \"Failed to create image from input bytes\"}";
            return resultStr.c_str();
        }

        // Apply rotation if needed
        cv::Mat rotatedImage;
        bool wasRotated = false;
        int rotatedWidth = inputWidth;
        int rotatedHeight = inputHeight;

        if (rotationDegrees == 90) {
            cv::rotate(inputImage, rotatedImage, cv::ROTATE_90_CLOCKWISE);
            wasRotated = true;
            rotatedWidth = inputHeight;
            rotatedHeight = inputWidth;
        } else if (rotationDegrees == 270 || rotationDegrees == -90) {
            cv::rotate(inputImage, rotatedImage, cv::ROTATE_90_COUNTERCLOCKWISE);
            wasRotated = true;
            rotatedWidth = inputHeight;
            rotatedHeight = inputWidth;
        } else if (rotationDegrees == 180) {
            cv::rotate(inputImage, rotatedImage, cv::ROTATE_180);
            wasRotated = true;
            rotatedWidth = inputWidth;
            rotatedHeight = inputHeight;
        } else {
            rotatedImage = inputImage;
        }

        LOGI("[normalize_image_bgra] After rotation: %dx%d, wasRotated: %s", rotatedWidth,
             rotatedHeight, wasRotated ? "true" : "false");

        // Calculate 16:9 landscape canvas dimensions
        // Use the larger dimension as the height reference
        const double TARGET_ASPECT_RATIO = 16.0 / 9.0;
        int canvasHeight = std::max(rotatedWidth, rotatedHeight);
        int canvasWidth = static_cast<int>(std::round(canvasHeight * TARGET_ASPECT_RATIO));

        LOGI("[normalize_image_bgra] Canvas dimensions: %dx%d", canvasWidth, canvasHeight);

        // Check if output buffer is large enough
        int requiredSize = canvasWidth * canvasHeight * 4;
        if (outputBufferSize < requiredSize) {
            LOGE("[normalize_image_bgra] Output buffer too small: %d < %d", outputBufferSize,
                 requiredSize);
            resultStr = "{\"error\": \"Output buffer too small\"}";
            return resultStr.c_str();
        }

        // Create black canvas (BGRA format)
        cv::Mat canvas(canvasHeight, canvasWidth, CV_8UC4, cv::Scalar(0, 0, 0, 255));

        // Calculate centered position
        int offsetX = (canvasWidth - rotatedWidth) / 2;
        int offsetY = (canvasHeight - rotatedHeight) / 2;

        LOGI("[normalize_image_bgra] Image offset on canvas: (%d, %d)", offsetX, offsetY);

        // Copy rotated image onto canvas at centered position
        cv::Rect roi(offsetX, offsetY, rotatedWidth, rotatedHeight);
        rotatedImage.copyTo(canvas(roi));

        // Copy canvas data to output buffer
        // Canvas is already in BGRA format with dense stride
        if (canvas.isContinuous()) {
            std::memcpy(outputBytes, canvas.data, requiredSize);
        } else {
            // Copy row by row if not continuous
            for (int y = 0; y < canvasHeight; ++y) {
                std::memcpy(outputBytes + y * canvasWidth * 4, canvas.ptr(y), canvasWidth * 4);
            }
        }

        LOGI("[normalize_image_bgra] Normalization complete");

        // Build JSON result with metadata
        std::string json = "{";
        json += "\"width\": " + std::to_string(canvasWidth) + ", ";
        json += "\"height\": " + std::to_string(canvasHeight) + ", ";
        json += "\"stride\": " + std::to_string(canvasWidth * 4) + ", ";
        json += "\"offset_x\": " + std::to_string(offsetX) + ", ";
        json += "\"offset_y\": " + std::to_string(offsetY) + ", ";
        json += "\"was_rotated\": " + std::string(wasRotated ? "true" : "false");
        json += "}";

        resultStr = json;
        return resultStr.c_str();

    } catch (const std::exception& e) {
        LOGE("[normalize_image_bgra] Exception: %s", e.what());
        resultStr = "{\"error\": \"" + std::string(e.what()) + "\"}";
        return resultStr.c_str();
    }
}
}
