#include <algorithm>
#include <iostream>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <vector>

#ifdef BUILD_SHARED_LIB
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#else
#include <onnxruntime/onnxruntime_cxx_api.h>
#endif

#include "ball_detector.hpp"
#include "table_detector.hpp"
#include "utilities.hpp"

using namespace std;
using namespace cv;

#define CONF_THRESH 0.6
#define IOU_THRESH 0.5

int runTableizerForImage(Mat image, BallDetector& ballDetector) {
#if LOCAL_BUILD
    imshow("Table", image);
    waitKey(0);
#endif
    // 2. Detect Table
    int cellSize = 20;
    double deltaEThreshold = 20.0;
    CellularTableDetector tableDetector(image.rows, cellSize, deltaEThreshold);

    cout << "--- 2: Table Detection ---" << endl;
    cout << "Parameters:  cellSize=" << cellSize << ", deltaEThreshold=" << deltaEThreshold << endl;

    Mat mask, tableDetection;

    tableDetector.detect(image, mask, tableDetection);
    cout << "Resized image for detection dimensions: " << tableDetection.cols << "x"
         << tableDetection.rows << endl;

    std::vector<cv::Point2f> quadPoints =
        tableDetector.quadFromInside(mask, tableDetection.cols, tableDetection.rows);

#if LOCAL_BUILD
    std::vector<cv::Point> quadDraw;
    quadDraw.reserve(quadPoints.size());

    for (const auto& pt : quadPoints) quadDraw.emplace_back(cvRound(pt.x), cvRound(pt.y));

    cv::polylines(tableDetection, quadDraw, true, cv::Scalar(0, 0, 255), 5);
    imshow("Quad Found", tableDetection);
    waitKey(0);
#endif
    if (quadPoints.size() != 4) {
        cerr << "Error: Could not detect table quad." << endl;
        return -1;
    }

#if LOCAL_BUILD
    cout << "Detected table quad corners (in resized image coordinates):" << endl;
    for (const auto& p : quadPoints) {
        cout << "  - (" << p.x << ", " << p.y << ")" << endl;
    }
    cout << endl;
#endif

    // 3. Warp Table
    bool rotate = true;
    WarpResult warpResult = warpTable(image, quadPoints, "warp.jpg", 840, rotate);

    // --- Ball detection & drawing --------------------------
    // 4. Detect balls **on the original image**
    cout << "--- Step 3: Ball Detection ---" << endl;
    const vector<Detection> detections = ballDetector.detect(image, CONF_THRESH, IOU_THRESH);
    cout << "Found " << detections.size() << " balls after non-maximum suppression.\n\n";

    // 5. Build transform: original-pixel  ➜  table_detection  ➜  canonical table
    // --------------------------------------------------------
    // table_detector resized the image to `resizeHeight` while preserving aspect.
    // Derive the scale used for that resize so we can link both spaces together.
    const double scaleY =
        static_cast<double>(tableDetection.rows) / static_cast<double>(image.rows);
    const double scaleX =
        static_cast<double>(tableDetection.cols) / static_cast<double>(image.cols);

    // homogeneous scale matrix (3×3)
    cv::Mat Hscale = (cv::Mat_<double>(3, 3) << scaleX, 0, 0, 0, scaleY, 0, 0, 0, 1);

    // Make sure both matrices share the same depth
    cv::Mat Hwarp;
    warpResult.transform.convertTo(Hwarp, CV_64F);  // canonical ← resized
    cv::Mat Htotal = Hscale * Hwarp;                // canonical ← original

    // 6. Draw predictions on the canonical table and shot-studio template
    // --------------------------------------------------------
#if LOCAL_BUILD
    string studioPath =
        "/Users/uzbit/Documents/projects/tableizer/data/shotstudio_table_felt_only.png";
    cv::Mat shotStudio = cv::imread(studioPath);
#endif
    cv::Mat warpedOut = warpResult.warped.clone();  // copy for drawing
    const cv::Scalar textColor(255, 255, 255);      // white id

    if (detections.empty()) {
        cout << "No balls detected.\n\n";
    } else {
        // centres in *original* pixel space
        vector<cv::Point2f> ballCentresOrig;
        for (const auto& d : detections) {
            cv::Point2f p = d.center;  //(d.box.x + d.box.width / 2, d.box.y + d.box.height / 2);
            ballCentresOrig.emplace_back(p);
            cout << "Ball at @ " << p << "\n";
        }

        // map straight to canonical table space
        vector<cv::Point2f> ballCentresCanonical;
        cv::perspectiveTransform(ballCentresOrig, ballCentresCanonical, Htotal);

        // --- Compute pixel radius based on table size ---
        int tableSizeInches = 100;  // change to 78, 88 or 100 for larger tables
        int longEdgePx = std::max(warpedOut.cols, warpedOut.rows);
        int ballDiameterPx = std::max(int(round(longEdgePx * (2.25 / tableSizeInches))), 8);
        int radius = ballDiameterPx / 2;

        cout << "--- Step 4: Final Ball Locations ---\\n";
        float textSize = 0.7 * (radius / 8.0);  // scale font with ball size

        for (size_t i = 0; i < ballCentresCanonical.size(); ++i) {
            const auto& p = ballCentresCanonical[i];
            cout << "  • class " << detections[i].classId << " conf " << detections[i].confidence
                 << " @ (" << p.x << ", " << p.y << ")\n";

#if LOCAL_BUILD
            cv::Scalar ballColor;
            switch (detections[i].classId) {
                case 3:
                    ballColor = cv::Scalar(0, 0, 255);
                    break;  // stripe → red
                case 2:
                    ballColor = cv::Scalar(255, 222, 33);
                    break;  // solid → yellow
                case 1:
                    ballColor = cv::Scalar(255, 255, 255);
                    break;  // cue → white
                case 0:
                    ballColor = cv::Scalar(0, 0, 0);
                    break;  // black
                default:
                    ballColor = cv::Scalar(128, 128, 128);
                    break;  // fallback
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
#endif
        }
        cout << endl;
    }

#if LOCAL_BUILD
    // 7. Display results
    cv::imshow("Warped Table + Balls", warpedOut);
    if (!shotStudio.empty()) {
        cv::imshow("Shot-Studio Overlay", shotStudio);
    }
    cv::waitKey(0);
#endif

    return 0;
}

#if BUILD_SHARED_LIB
// #include <android/log.h>

#include "ball_detector.hpp"
#include "tableizer.hpp"

// #define LOG_TAG "tableizer"  // anything that helps you filter Logcat
// #define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
// #define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

// The FFI functions will interact with the BallDetector class.
// We return a void* to hide the implementation details from the C interface.

string format_detections_json(const vector<Detection>& detections) {
    // Format results as a JSON string
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
        BallDetector* detector = new BallDetector(model_path);
        return static_cast<void*>(detector);
    } catch (const std::exception& e) {
        std::cerr << "Error initializing detector: " << e.what() << std::endl;
        return nullptr;
    }
}

const char* detect_objects_rgba(void* detector_ptr, const unsigned char* image_bytes, int width,
                                int height, int channels) {
    // cv::setNumThreads(0);
    // cv::setUseOptimized(false);
    static std::string result_str;
    if (!detector_ptr) {
        result_str = "{\"error\": \"Invalid detector instance\"}";
        return result_str.c_str();
    }

    BallDetector* detector = static_cast<BallDetector*>(detector_ptr);

    try {
        // Create cv::Mat from raw bytes. We assume RGBA (4 channels) from Flutter.
        cv::Mat image(height, width, CV_8UC4, (void*)image_bytes);
        if (image.empty()) {
            result_str = "{\"error\": \"Failed to create image from bytes\"}";
            return result_str.c_str();
        }

        // Convert to BGR for processing if needed by the model
        cv::Mat image_bgr;
        cv::cvtColor(image, image_bgr, cv::COLOR_RGBA2BGR);

        const auto detections = detector->detect(image_bgr, CONF_THRESH, IOU_THRESH);

        result_str = format_detections_json(detections);
        // LOGI(result_str);
        return result_str.c_str();

    } catch (const std::exception& e) {
        result_str = std::string("{\"error\": \"") + e.what() + "\"}";
        // LOGE(result_str);
        return result_str.c_str();
    }
}

void release_detector(void* detector_ptr) {
    if (detector_ptr) {
        BallDetector* detector = static_cast<BallDetector*>(detector_ptr);
        delete detector;
    }
}

const char* detect_objects_yuv(void* detector_ptr, uint8_t* y_plane, uint8_t* u_plane,
                               uint8_t* v_plane, int width, int height, int y_stride, int u_stride,
                               int v_stride) {
    static std::string result_str;
    if (!detector_ptr) {
        result_str = "{\"error\": \"Invalid detector instance\"}";
        return result_str.c_str();
    }

    BallDetector* detector = static_cast<BallDetector*>(detector_ptr);

    try {
        // Create a single continuous cv::Mat for the I420 data.
        cv::Mat yuv(height + height / 2, width, CV_8UC1);
        uint8_t* yuv_data = yuv.data;

        // Copy Y plane, handling stride.
        for (int i = 0; i < height; ++i) {
            memcpy(yuv_data + i * width, y_plane + i * y_stride, width);
        }

        // Copy U plane, handling stride.
        uint8_t* u_dst = yuv_data + height * width;
        for (int i = 0; i < height / 2; ++i) {
            memcpy(u_dst + i * (width / 2), u_plane + i * u_stride, width / 2);
        }

        // Copy V plane, handling stride.
        uint8_t* v_dst = u_dst + (height / 2) * (width / 2);
        for (int i = 0; i < height / 2; ++i) {
            memcpy(v_dst + i * (width / 2), v_plane + i * v_stride, width / 2);
        }

        // Convert the packed YUV image to BGR.
        cv::Mat bgr;
        cv::cvtColor(yuv, bgr, cv::COLOR_YUV2BGR_I420);
        const auto detections = detector->detect(bgr, CONF_THRESH, IOU_THRESH);

        result_str = format_detections_json(detections);
        return result_str.c_str();

    } catch (const std::exception& e) {
        result_str = std::string("{\"error\": \"") + e.what() + "\"}";
        return result_str.c_str();
    }
}

const char* detect_table_yuv(uint8_t* y_plane, uint8_t* u_plane, uint8_t* v_plane, int width,
                             int height, int y_stride, int u_stride, int v_stride) {
    static std::string result_str;

    try {
        // Create a single continuous cv::Mat for the I420 data.
        cv::Mat yuv(height + height / 2, width, CV_8UC1);
        uint8_t* yuv_data = yuv.data;

        // Copy Y plane, handling stride.
        for (int i = 0; i < height; ++i) {
            memcpy(yuv_data + i * width, y_plane + i * y_stride, width);
        }

        // Copy U plane, handling stride.
        uint8_t* u_dst = yuv_data + height * width;
        for (int i = 0; i < height / 2; ++i) {
            memcpy(u_dst + i * (width / 2), u_plane + i * u_stride, width / 2);
        }

        // Copy V plane, handling stride.
        uint8_t* v_dst = u_dst + (height / 2) * (width / 2);
        for (int i = 0; i < height / 2; ++i) {
            memcpy(v_dst + i * (width / 2), v_plane + i * v_stride, width / 2);
        }

        // Convert the packed YUV image to BGR.
        cv::Mat bgr;
        cv::cvtColor(yuv, bgr, cv::COLOR_YUV2BGR_I420);

        // Detect table
        int cellSize = 20;
        double deltaEThreshold = 20.0;
        CellularTableDetector tableDetector(bgr.rows, cellSize, deltaEThreshold);
        Mat mask, tableDetection;
        tableDetector.detect(bgr, mask, tableDetection);
        std::vector<cv::Point2f> quadPoints =
            tableDetector.quadFromInside(mask, tableDetection.cols, tableDetection.rows);

        // Format results as a JSON string
        std::string json = "{\"quad_points\": [";
        for (size_t i = 0; i < quadPoints.size(); ++i) {
            json += "{\"x\": " + std::to_string(quadPoints[i].x) +
                    ", \"y\": " + std::to_string(quadPoints[i].y) + "}";
            if (i < quadPoints.size() - 1) {
                json += ", ";
            }
        }
        json += "]}";

        result_str = json;
        return result_str.c_str();

    } catch (const std::exception& e) {
        result_str = std::string("{\"error\": \"") + e.what() + "\"}";
        return result_str.c_str();
    }
}

const char* detect_table_rgba(const unsigned char* image_bytes, int width, int height,
                              int channels) {
    // cv::setNumThreads(0);
    // cv::setUseOptimized(false);
    static std::string result_str;

    try {
        // Create cv::Mat from raw bytes. We assume RGBA (4 channels) from Flutter.
        cv::Mat image(height, width, CV_8UC4, (void*)image_bytes);
        if (image.empty()) {
            result_str = "{\"error\": \"Failed to create image from bytes\"}";
            return result_str.c_str();
        }

        // Convert to BGR for processing if needed by the model
        cv::Mat bgr;
        cv::cvtColor(image, bgr, cv::COLOR_RGBA2BGR);

        // Detect table
        int cellSize = 20;
        double deltaEThreshold = 20.0;
        CellularTableDetector tableDetector(bgr.rows, cellSize, deltaEThreshold);
        Mat mask, tableDetection;
        tableDetector.detect(bgr, mask, tableDetection);
        std::vector<cv::Point2f> quadPoints =
            tableDetector.quadFromInside(mask, tableDetection.cols, tableDetection.rows);

        // Format results as a JSON string
        std::string json = "{\"quad_points\": [";
        for (size_t i = 0; i < quadPoints.size(); ++i) {
            json += "{\"x\": " + std::to_string(quadPoints[i].x) +
                    ", \"y\": " + std::to_string(quadPoints[i].y) + "}";
            if (i < quadPoints.size() - 1) {
                json += ", ";
            }
        }
        json += "]}";

        result_str = json;
        return result_str.c_str();

    } catch (const std::exception& e) {
        result_str = std::string("{\"error\": \"") + e.what() + "\"}";
        return result_str.c_str();
    }
}

}  // extern C
#endif
