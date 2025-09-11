#include <algorithm>
#include <iostream>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <vector>

// Include all project headers at the top
#include "ball_detector.hpp"
#include "table_detector.hpp"
#include "tableizer.hpp"
#include "utilities.hpp"
#include "base64_utils.hpp"

using namespace std;
using namespace cv;

// Ball detection constants
#define CONF_THRESH 0.6
#define IOU_THRESH 0.5

// Table detection constants
#define CELL_SIZE 16  // Table detection, determines how fine/corse detection (affects fps).
#define DELTAE_THRESH \
    25.0  // Table detection, determines how close the median color of a cell must be to the target
          // color for inclusion in mask.
#define RESIZE_STREAM 864   // Use for streaming detection only
#define RESIZE_STATIC 1680  // Use for static/high-quality detection

int runTableizerForImage(Mat image, BallDetector& ballDetector) {
#ifdef PLATFORM_MACOS
    imshow("Table", image);
    waitKey(0);

    // --- 1. Get quad points via FFI call ---
    LOGI("--- 1: Table Detection (FFI) ---");
    cv::Mat bgra_image;
    cv::cvtColor(image, bgra_image, cv::COLOR_BGR2BGRA);
    const char* jsonResult = detect_table_bgra(bgra_image.data, bgra_image.cols, bgra_image.rows,
                                               bgra_image.step, 0, nullptr);

    if (jsonResult == nullptr) {
        LOGE("Error: FFI detection returned null.");
        return -1;
    }

    // Parse JSON result (simple manual parsing for test code)
    std::string jsonStr(jsonResult);
    std::vector<cv::Point2f> ffiQuadPoints;
    cv::Mat quadMask;

    // Find quad_points array in JSON - now using [x,y] format
    size_t quadStart = jsonStr.find("\"quad_points\":");
    if (quadStart == std::string::npos) {
        LOGE("Error: No quad_points found in JSON response.");
        return -1;
    }

    // Extract quad points from new [x,y] format
    size_t arrayStart = jsonStr.find("[", quadStart);
    if (arrayStart != std::string::npos) {
        size_t pos = arrayStart + 1;
        int pointCount = 0;
        
        while (pos < jsonStr.length() && pointCount < 4) {
            // Find start of next point array [x,y]
            size_t pointStart = jsonStr.find("[", pos);
            if (pointStart == std::string::npos) break;
            
            size_t pointEnd = jsonStr.find("]", pointStart);
            if (pointEnd == std::string::npos) break;
            
            // Extract point values
            std::string pointStr = jsonStr.substr(pointStart + 1, pointEnd - pointStart - 1);
            size_t comma = pointStr.find(",");
            if (comma != std::string::npos) {
                float x = std::stof(pointStr.substr(0, comma));
                float y = std::stof(pointStr.substr(comma + 1));
                ffiQuadPoints.emplace_back(x, y);
                pointCount++;
            }
            
            pos = pointEnd + 1;
        }
    }
    
    // Parse mask data if present
    size_t maskStart = jsonStr.find("\"mask\":");
    if (maskStart != std::string::npos) {
        // Extract mask base64 data
        size_t dataStart = jsonStr.find("\"data\":", maskStart);
        if (dataStart != std::string::npos) {
            size_t dataValueStart = jsonStr.find("\"", dataStart + 7) + 1;
            size_t dataValueEnd = jsonStr.find("\"", dataValueStart);
            if (dataValueEnd != std::string::npos) {
                std::string maskBase64 = jsonStr.substr(dataValueStart, dataValueEnd - dataValueStart);
                
                // Decode base64 mask using existing utilities
                LOGI("Found mask data (%zu bytes base64)", maskBase64.length());
                try {
                    std::vector<unsigned char> maskData = Base64Utils::decode(maskBase64);
                    if (!maskData.empty()) {
                        // Decode PNG mask data
                        quadMask = cv::imdecode(maskData, cv::IMREAD_GRAYSCALE);
                        if (!quadMask.empty()) {
                            LOGI("Successfully decoded mask: %dx%d", quadMask.cols, quadMask.rows);
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

    // --- Continue with the rest of the process using the FFI points ---
    // FFI now returns coordinates already scaled to original image dimensions
    std::vector<cv::Point2f> quadPoints = ffiQuadPoints;

    // Draw quad on original image using scaled coordinates
    std::vector<cv::Point> quadDraw;
    quadDraw.reserve(quadPoints.size());
    for (const auto& pt : quadPoints) quadDraw.emplace_back(cvRound(pt.x), cvRound(pt.y));
    Mat imageWithQuad = image.clone();
    cv::polylines(imageWithQuad, quadDraw, true, cv::Scalar(0, 0, 255), 5);

#ifdef PLATFORM_MACOS
    imshow("Quad Found (from FFI)", imageWithQuad);
    waitKey(0);
#endif

    // 3. Warp Table
    // Compute rotation like Dart code: check if topLength > rightLength * TABLE_ASPECT_RATIO_THRESHOLD
    // Quad points are already ordered by orderQuad function
    float topLength = cv::norm(quadPoints[1] - quadPoints[0]);    // top edge
    float rightLength = cv::norm(quadPoints[2] - quadPoints[1]);  // right edge
    bool rotate = topLength < rightLength * 1.75;
    LOGI("Quad analysis: topLength=%.2f, rightLength=%.2f, rotate=%s", topLength, rightLength,
         rotate ? "true" : "false");
    WarpResult warpResult = warpTable(image, quadPoints, "warp.jpg", 864, rotate);

    // --- Ball detection & drawing --------------------------
    // 4. Detect balls **on the original image** (with mask if available)
    LOGI("--- Step 4: Ball Detection ---");
    
    // Apply mask to image if available
    Mat imageForDetection = image;
    if (!quadMask.empty()) {
        // Resize mask to match original image dimensions if needed
        Mat maskResized;
        if (quadMask.size() != image.size()) {
            cv::resize(quadMask, maskResized, image.size());
        } else {
            maskResized = quadMask;
        }
        
        // Apply mask - zero out non-table regions
        cv::bitwise_and(image, image, imageForDetection, maskResized);
        LOGI("Applied table mask to image for ball detection");
    } else {
        LOGI("No mask available, using original image for ball detection");
    }
    
    const vector<Detection> detections = ballDetector.detect(imageForDetection, CONF_THRESH, IOU_THRESH);
    LOGI("Found %zu balls after non-maximum suppression.", detections.size());

    // 5. Build transform: original-pixel  ➜  canonical table
    // --------------------------------------------------------
    // Since quad points are now scaled to original image coordinates,
    // we can use the warp transform directly without additional scaling
    cv::Mat Hwarp;
    warpResult.transform.convertTo(Hwarp, CV_64F);
    cv::Mat Htotal = Hwarp;

    // 6. Draw predictions on the canonical table and shot-studio template
    // --------------------------------------------------------
    cv::Mat shotStudio;
#if defined(PLATFORM_MACOS) || defined(PLATFORM_LINUX) || defined(PLATFORM_WINDOWS)
    string studioPath =
        "/Users/uzbit/Documents/projects/tableizer/data/shotstudio_table_felt_only.png";
    shotStudio = cv::imread(studioPath);
#endif
    cv::Mat warpedOut = warpResult.warped.clone();  // copy for drawing
    const cv::Scalar textColor(255, 255, 255);      // white id

    if (detections.empty()) {
        LOGI("No balls detected.");
    } else {
        // centres in *original* pixel space
        vector<cv::Point2f> ballCentresOrig;
        for (const auto& d : detections) {
            ballCentresOrig.emplace_back(d.center);
        }

        // map straight to canonical table space
        vector<cv::Point2f> ballCentresCanonical;
        cv::perspectiveTransform(ballCentresOrig, ballCentresCanonical, Htotal);

        // --- Compute pixel radius based on table size ---
        int tableSizeInches = 100;
        int longEdgePx = std::max(warpedOut.cols, warpedOut.rows);
        int radius = std::max((int)round(longEdgePx * (2.25 / tableSizeInches) / 2.0), 4);
        float textSize = 0.7 * (radius / 8.0);

        for (size_t i = 0; i < ballCentresCanonical.size(); ++i) {
            const auto& p = ballCentresCanonical[i];
            LOGI("  • class %d conf %.3f @ (%.1f, %.1f)", detections[i].classId,
                 detections[i].confidence, p.x, p.y);

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

            // Draw on warpedOut
            cv::circle(warpedOut, p, radius, ballColor, cv::FILLED, cv::LINE_AA);
            cv::putText(warpedOut, std::to_string(detections[i].classId),
                        p + cv::Point2f(radius + 2, 0), cv::FONT_HERSHEY_SIMPLEX, textSize,
                        textColor, 2);

            // Draw on canonical studio
            if (!shotStudio.empty()) {
                cv::circle(shotStudio, p, radius, ballColor, cv::FILLED);
                cv::putText(shotStudio, std::to_string(detections[i].classId),
                            p + cv::Point2f(radius + 2, 0), cv::FONT_HERSHEY_SIMPLEX, textSize,
                            textColor, 2);
            }
        }
    }

    // 7. Display results
#ifdef PLATFORM_MACOS
    cv::imshow("Warped Table + Balls", warpedOut);
    if (!shotStudio.empty()) {
        cv::imshow("Shot-Studio Overlay", shotStudio);
    }
    cv::waitKey(0);
#endif
#endif

    return 0;
}

string format_detections_json(const vector<Detection>& detections) {
    std::string json = "{\"detections\": [";
    for (size_t i = 0; i < detections.size(); ++i) {
        const auto& d = detections[i];
        json += "{";
        json += "\"class_id\": " + std::to_string(d.classId) + ", ";
        json += "\"confidence\": " + std::to_string(d.confidence) + ", ";
        json += "\"center_x\": " + std::to_string(d.center.x) + ", ";
        json += "\"center_y\": " + std::to_string(d.center.y) + ", ";
        json += "\"box\": {\"x\": " + std::to_string(d.box.x) +
                ", \"y\": " + std::to_string(d.box.y) +
                ", \"width\": " + std::to_string(d.box.width) +
                ", \"height\": " + std::to_string(d.box.height) + "}";
        json += "}";
        if (i < detections.size() - 1) {
            json += ", ";
        }
    }
    json += "]}";
    return json;
}

extern "C" {

void* initialize_detector(const char* model_path) {
    try {
        LOGI("initialize_detector called with path: %s", model_path ? model_path : "NULL");
        BallDetector* detector = new BallDetector(model_path);
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


const char* detect_objects_bgra(void* detector_ptr, const unsigned char* image_bytes, int width,
                                int height, int stride) {
    static std::string result_str;
    if (!detector_ptr) {
        result_str = "{\"error\": \"Invalid detector instance\"}";
        return result_str.c_str();
    }
    try {
        cv::Mat bgra_image(height, width, CV_8UC4, (void*)image_bytes, stride);
        if (bgra_image.empty()) {
            result_str = "{\"error\": \"Failed to create image from bytes\"}";
            return result_str.c_str();
        }
        cv::Mat image_bgr;
        cv::cvtColor(bgra_image, image_bgr, cv::COLOR_BGRA2BGR);
        const auto detections =
            static_cast<BallDetector*>(detector_ptr)->detect(image_bgr, CONF_THRESH, IOU_THRESH);
        result_str = format_detections_json(detections);
        return result_str.c_str();
    } catch (const std::exception& e) {
        result_str = std::string("{\"error\": \"") + e.what() + "\"}";
        return result_str.c_str();
    }
}

void release_detector(void* detector_ptr) {
    if (detector_ptr) {
        delete static_cast<BallDetector*>(detector_ptr);
    }
}

const char* detect_table_bgra(const unsigned char* image_bytes, int width, int height, int stride,
                              int rotation_degrees, const char* debug_image_path) {
    static std::string result_str;
    try {
        cv::Mat bgra_image_unrotated(height, width, CV_8UC4, (void*)image_bytes, stride);
        if (bgra_image_unrotated.empty()) {
            LOGE("Failed to create image from bytes.");
            result_str = "{\"error\": \"Failed to create image from bytes\"}";
            return result_str.c_str();
        }

        CellularTableDetector tableDetector(RESIZE_STREAM, CELL_SIZE, DELTAE_THRESH);
        Mat cellularMask, tableDetection;
        tableDetector.detect(bgra_image_unrotated, cellularMask, tableDetection, 0);

        std::vector<cv::Point2f> quadPoints = tableDetector.getQuadFromMask(cellularMask);
        
        // Scale quad points back to original image size
        // The detector resizes the image to RESIZE_STREAM (864) pixels height, so we need to scale back
        float scaleX = (float)bgra_image_unrotated.cols / tableDetection.cols;
        float scaleY = (float)bgra_image_unrotated.rows / tableDetection.rows;
        
        std::vector<cv::Point2f> scaledQuadPoints;
        for (const auto& pt : quadPoints) {
            scaledQuadPoints.emplace_back(pt.x * scaleX, pt.y * scaleY);
        }
        
        // Create clean quadrilateral mask instead of cellular detection mask
        Mat quadMask = Mat::zeros(bgra_image_unrotated.size(), CV_8UC1);
        if (scaledQuadPoints.size() == 4) {
            // Convert scaled Point2f to Point for fillPoly
            std::vector<cv::Point> quadPointsInt;
            for (const auto& pt : scaledQuadPoints) {
                quadPointsInt.emplace_back(cvRound(pt.x), cvRound(pt.y));
            }
            
            // Fill the quadrilateral with white (255)
            std::vector<std::vector<cv::Point>> contours = {quadPointsInt};
            cv::fillPoly(quadMask, contours, cv::Scalar(255));
        }

        /*
        // --- Draw Quad on Debug Image (DISABLED FOR PERFORMANCE) ---
        if (debug_image_path != nullptr && strlen(debug_image_path) > 0) {
            if (quadPoints.size() == 4) {
                std::vector<cv::Point> quadDraw;
                quadDraw.reserve(quadPoints.size());
                for (const auto& pt : quadPoints) {
                    quadDraw.emplace_back(cvRound(pt.x), cvRound(pt.y));
                }
                cv::polylines(tableDetection, quadDraw, true, cv::Scalar(0, 0, 255), 5);
            }
            LOGI("Attempting to save debug image to: %s", debug_image_path);
            Mat rgb;
            cv::cvtColor(tableDetection, rgb, cv::COLOR_BGRA2RGB);
            if (cv::imwrite(debug_image_path, rgb)) {
                LOGI("Successfully saved debug image with quad overlay.");
            } else {
                LOGE("Failed to save debug image.");
            }
        }
        */

        // Encode mask as base64 PNG
        std::string maskBase64 = Base64Utils::encodeMat(quadMask);
        
        std::string json = "{\"quad_points\": [";
        for (size_t i = 0; i < scaledQuadPoints.size(); ++i) {
            json += "[" + std::to_string(scaledQuadPoints[i].x) +
                    ", " + std::to_string(scaledQuadPoints[i].y) + "]";
            if (i < scaledQuadPoints.size() - 1) json += ", ";
        }
        json += "], \"image_width\": " + std::to_string(bgra_image_unrotated.cols) +
                ", \"image_height\": " + std::to_string(bgra_image_unrotated.rows);
        
        // Add mask data if encoding was successful
        if (!maskBase64.empty()) {
            json += ", \"mask\": {";
            json += "\"width\": " + std::to_string(quadMask.cols) + ",";
            json += "\"height\": " + std::to_string(quadMask.rows) + ",";
            json += "\"data\": \"" + maskBase64 + "\"";
            json += "}";
        }
        
        json += "}";

        result_str = json;
        return result_str.c_str();

    } catch (const std::exception& e) {
        LOGE("Error in detect_table_bgra: %s", e.what());
        result_str =
            "{\"error\": \"Exception in detect_table_bgra: " + std::string(e.what()) + "\"}";
        return result_str.c_str();
    }
}

void free_bgra_detection_result(DetectionResult* result) {
    if (result) {
        delete result;
    }
}


// Coordinate transformation FFI exports
const char* transform_points_using_quad(const float* points_data, int points_count,
                                        const float* quad_data, int quad_count, int image_width,
                                        int image_height, int display_width, int display_height) {
    static std::string result_str;

    try {
        if (points_count == 0 || quad_count != 4) {
            result_str = "{\"error\": \"Invalid input parameters\"}";
            return result_str.c_str();
        }

        // Convert input data to OpenCV points
        std::vector<cv::Point2f> points;
        for (int i = 0; i < points_count; i++) {
            points.emplace_back(points_data[i * 2], points_data[i * 2 + 1]);
        }

        std::vector<cv::Point2f> quadPoints;
        for (int i = 0; i < quad_count; i++) {
            quadPoints.emplace_back(quad_data[i * 2], quad_data[i * 2 + 1]);
        }

        // Call the transformation function
        TransformationResult transformation =
            transformPointsUsingQuad(points, quadPoints, cv::Size(image_width, image_height),
                                     cv::Size(display_width, display_height));

        // Format result as JSON
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
        json += "}";

        result_str = json;
        return result_str.c_str();

    } catch (const std::exception& e) {
        result_str = "{\"error\": \"" + std::string(e.what()) + "\"}";
        return result_str.c_str();
    }
}

}  // extern C