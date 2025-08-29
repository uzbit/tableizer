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

#if defined(PLATFORM_ANDROID)
#include <core/session/onnxruntime_cxx_api.h>
#else
#include <onnxruntime_cxx_api.h>
#endif

using namespace std;
using namespace cv;

#define CONF_THRESH 0.6
#define IOU_THRESH 0.5
#define CELL_SIZE 10
#define DELTAE_THRESH 25.0
#define RESIZE 800  // not used

int runTableizerForImage(Mat image, BallDetector& ballDetector) {
#if defined(LOCAL_BUILD) && \
    (defined(PLATFORM_MACOS) || defined(PLATFORM_LINUX) || defined(PLATFORM_WINDOWS))
    imshow("Table", image);
    waitKey(0);

    // --- 1. Get ground truth quad points by calling the detector directly ---
    cout << "--- 1: Table Detection (Direct) ---" << endl;
    CellularTableDetector tableDetector(image.rows, CELL_SIZE, DELTAE_THRESH);
    Mat mask, tableDetection;
    tableDetector.detect(image, mask, tableDetection, 0);
    std::vector<cv::Point2f> directQuadPoints =
        tableDetector.quadFromInside(mask, tableDetection.cols, tableDetection.rows);

    if (directQuadPoints.size() != 4) {
        cerr << "Error: Direct detection failed to find 4 points." << endl;
        return -1;
    }

    // --- 2. Get quad points via FFI call ---
    cout << "--- 2: Table Detection (FFI) ---" << endl;
    cv::Mat bgra_image;
    cv::cvtColor(image, bgra_image, cv::COLOR_BGR2BGRA);
    DetectionResult* ffiResult = detect_table_bgra(bgra_image.data, bgra_image.cols,
                                                   bgra_image.rows, bgra_image.step, 0, nullptr);

    if (ffiResult == nullptr || ffiResult->quad_points_count != 4) {
        cerr << "Error: FFI detection failed to find 4 points." << endl;
        if (ffiResult) free_bgra_detection_result(ffiResult);
        return -1;
    }
    std::vector<cv::Point2f> ffiQuadPoints;
    for (int i = 0; i < ffiResult->quad_points_count; ++i) {
        ffiQuadPoints.emplace_back(ffiResult->quad_points[i].x, ffiResult->quad_points[i].y);
    }
    free_bgra_detection_result(ffiResult);

    // --- 3. Compare the results ---
    cout << "--- 3: Comparing Direct vs. FFI Results Table Quad ---" << endl;
    bool match = true;
    double epsilon = 1e-5;
    if (directQuadPoints.size() != ffiQuadPoints.size()) {
        match = false;
    } else {
        for (size_t i = 0; i < directQuadPoints.size(); ++i) {
            printf("x, y: %f, %f\n", directQuadPoints[i].x, directQuadPoints[i].y);

            if (std::abs(directQuadPoints[i].x - ffiQuadPoints[i].x) > epsilon ||
                std::abs(directQuadPoints[i].y - ffiQuadPoints[i].y) > epsilon) {
                match = false;
                break;
            }
        }
    }

    if (match) {
        cout << "SUCCESS: Quad points from direct call and FFI call match." << endl;
    } else {
        cerr << "FAILURE: Quad points do not match!" << endl;
        cerr << "  Direct Points:" << endl;
        for (const auto& p : directQuadPoints) cerr << "    (" << p.x << ", " << p.y << ")" << endl;
        cerr << "  FFI Points:" << endl;
        for (const auto& p : ffiQuadPoints) cerr << "    (" << p.x << ", " << p.y << ")" << endl;
        return -1;  // Exit on failure
    }
    cout << endl;

    // --- Continue with the rest of the process using the validated points ---
    std::vector<cv::Point2f> quadPoints = ffiQuadPoints;  // Use FFI points for subsequent steps

    std::vector<cv::Point> quadDraw;
    quadDraw.reserve(quadPoints.size());
    for (const auto& pt : quadPoints) quadDraw.emplace_back(cvRound(pt.x), cvRound(pt.y));
    cv::polylines(tableDetection, quadDraw, true, cv::Scalar(0, 0, 255), 5);
    imshow("Quad Found (from FFI)", tableDetection);
    waitKey(0);

    // 3. Warp Table
    bool rotate = true;
    WarpResult warpResult = warpTable(image, quadPoints, "warp.jpg", 840, rotate);

    // --- Ball detection & drawing --------------------------
    // 4. Detect balls **on the original image**
    cout << "--- Step 4: Ball Detection ---" << endl;
    const vector<Detection> detections = ballDetector.detect(image, CONF_THRESH, IOU_THRESH);
    cout << "Found " << detections.size() << " balls after non-maximum suppression.\n\n";

    // 5. Build transform: original-pixel  ➜  table_detection  ➜  canonical table
    // --------------------------------------------------------
    const double scaleY =
        static_cast<double>(tableDetection.rows) / static_cast<double>(image.rows);
    const double scaleX =
        static_cast<double>(tableDetection.cols) / static_cast<double>(image.cols);

    cv::Mat Hscale = (cv::Mat_<double>(3, 3) << scaleX, 0, 0, 0, scaleY, 0, 0, 0, 1);
    cv::Mat Hwarp;
    warpResult.transform.convertTo(Hwarp, CV_64F);
    cv::Mat Htotal = Hscale * Hwarp;

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
        cout << "No balls detected.\n\n";
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
            cout << "  • class " << detections[i].classId << " conf " << detections[i].confidence
                 << " @ (" << p.x << ", " << p.y << ")\n";

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
        cout << endl;
    }

    // 7. Display results
    cv::imshow("Warped Table + Balls", warpedOut);
    if (!shotStudio.empty()) {
        cv::imshow("Shot-Studio Overlay", shotStudio);
    }
    cv::waitKey(0);
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

const char* detect_objects_rgba(void* detector_ptr, const unsigned char* image_bytes, int width,
                                int height, int channels) {
    static std::string result_str;
    if (!detector_ptr) {
        result_str = "{\"error\": \"Invalid detector instance\"}";
        return result_str.c_str();
    }
    try {
        cv::Mat image(height, width, CV_8UC4, (void*)image_bytes);
        if (image.empty()) {
            result_str = "{\"error\": \"Failed to create image from bytes\"}";
            return result_str.c_str();
        }
        cv::Mat image_bgr;
        cv::cvtColor(image, image_bgr, cv::COLOR_RGBA2BGR);
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

DetectionResult* detect_table_bgra(const unsigned char* image_bytes, int width, int height,
                                   int stride, int rotation_degrees, const char* debug_image_path) {
    try {
        cv::Mat bgra_image_unrotated(height, width, CV_8UC4, (void*)image_bytes, stride);
        if (bgra_image_unrotated.empty()) {
            LOGE("Failed to create image from bytes.");
            return nullptr;
        }

        // CellularTableDetector tableDetector(bgra_image_unrotated.rows, CELL_SIZE, DELTAE_THRESH);

        CellularTableDetector tableDetector(bgra_image_unrotated.rows, CELL_SIZE, DELTAE_THRESH);
        Mat mask, tableDetection;
        tableDetector.detect(bgra_image_unrotated, mask, tableDetection, rotation_degrees);

        std::vector<cv::Point2f> quadPoints =
            tableDetector.quadFromInside(mask, tableDetection.cols, tableDetection.rows);

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

        DetectionResult* result = new DetectionResult();
        result->quad_points_count = quadPoints.size();
        result->image_width = tableDetection.cols;
        result->image_height = tableDetection.rows;
        for (size_t i = 0; i < quadPoints.size(); ++i) {
            result->quad_points[i] = {quadPoints[i].x, quadPoints[i].y};
        }
        // LOGI("Detected %d quad points.", result->quad_points_count);
        return result;

    } catch (const std::exception& e) {
        LOGE("Error in detect_table_bgra: %s", e.what());
        return nullptr;
    }
}

void free_bgra_detection_result(DetectionResult* result) {
    if (result) {
        delete result;
    }
}

const char* detect_table_rgba(const unsigned char* image_bytes, int width, int height, int channels,
                              int stride) {
    static std::string result_str;
    try {
        cv::Mat image(height, width, CV_8UC4, (void*)image_bytes, stride);
        if (image.empty()) {
            result_str = "{\"error\": \"Failed to create image from bytes\"}";
            return result_str.c_str();
        }
        cv::Mat bgr;
        cv::cvtColor(image, bgr, cv::COLOR_BGRA2BGR);

        CellularTableDetector tableDetector(bgr.rows, CELL_SIZE, DELTAE_THRESH);
        Mat mask, tableDetection;
        tableDetector.detect(bgr, mask, tableDetection, 0);
        std::vector<cv::Point2f> quadPoints =
            tableDetector.quadFromInside(mask, tableDetection.cols, tableDetection.rows);

        std::vector<uchar> buf;
        cv::imencode(".jpg", tableDetection, buf);
        std::string base64_image = base64_encode(buf.data(), buf.size());

        std::string json = "{\"quad_points\": [";
        for (size_t i = 0; i < quadPoints.size(); ++i) {
            json += "{\"x\": " + std::to_string(quadPoints[i].x) +
                    ", \"y\": " + std::to_string(quadPoints[i].y) + "}";
            if (i < quadPoints.size() - 1) {
                json += ", ";
            }
        }
        json += "], \"image\": \"" + base64_image + "\"}";
        result_str = json;
        return result_str.c_str();

    } catch (const std::exception& e) {
        result_str = std::string("{\"error\": \"") + e.what() + "\"}";
        return result_str.c_str();
    }
}

}  // extern C