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

#define IMSHOW true
#define CONF_THRESH 0.6
#define IOU_THRESH 0.7

int runTableizerForImage(Mat image, BallDetector& ballDetector) {
#if IMSHOW
    imshow("Table", image);
    waitKey(0);
#endif
    // 2. Detect Table
    int resizeHeight = 3000;
    int cellSize = 20;
    double deltaEThreshold = 20.0;
    CellularTableDetector tableDetector(image.rows, cellSize, deltaEThreshold);

    cout << "--- 2: Table Detection ---" << endl;
    cout << "Parameters: resizeHeight=" << resizeHeight << ", cellSize=" << cellSize
         << ", deltaEThreshold=" << deltaEThreshold << endl;

    Mat mask, tableDetection;

    tableDetector.detect(image, mask, tableDetection);
    cout << "Resized image for detection dimensions: " << tableDetection.cols << "x"
         << tableDetection.rows << endl;

    std::vector<cv::Point2f> quadPoints =
        tableDetector.quadFromInside(mask, tableDetection.cols, tableDetection.rows);

#if IMSHOW
    std::vector<cv::Point> quadDraw;
    quadDraw.reserve(quadPoints.size());

    for (const auto& pt : quadPoints) quadDraw.emplace_back(cvRound(pt.x), cvRound(pt.y));

    cv::polylines(tableDetection, quadDraw, true, cv::Scalar(0, 0, 255), 5);
    imshow("Quad Found", tableDetection);
    waitKey(0);
#endif
    if (quadPoints.size() != 4) {
        cerr << "Error: Could not detect table quad." << endl;
        imshow("Debug: No Quad Found", tableDetection);
        waitKey(0);
        return -1;
    }

#if IMSHOW
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
    string studioPath =
        "/Users/uzbit/Documents/projects/tableizer/data/shotstudio_table_felt_only.png";
    cv::Mat shotStudio = cv::imread(studioPath);

    cv::Mat warpedOut = warpResult.warped.clone();  // copy for drawing
    const cv::Scalar textColor(255, 255, 255);      // white id

    if (detections.empty()) {
        cout << "No balls detected.\n\n";
    } else {
        // centres in *original* pixel space
        vector<cv::Point2f> ballCentresOrig;
        for (const auto& d : detections) {
            cv::Point2f p(d.box.x + d.box.width / 2, d.box.y + d.box.height / 2);
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

        cout << "--- Step 4: Final Ball Locations ---\n";
        float textSize = 0.7 * (radius / 8.0);  // scale font with ball size

        for (size_t i = 0; i < ballCentresCanonical.size(); ++i) {
            const auto& p = ballCentresCanonical[i];
            cout << "  • class " << detections[i].classId << " @ (" << p.x << ", " << p.y << ")\n";

#if IMSHOW
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

#if IMSHOW
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
#include "tableizer.hpp"

// The FFI functions will interact with the BallDetector class.
// We return a void* to hide the implementation details from the C interface.

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

const char* detect_objects(void* detector_ptr, const unsigned char* image_bytes, int width,
                           int height, int channels) {
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

        // Format results as a JSON string
        std::string json = "{\"detections\": [";
        for (size_t i = 0; i < detections.size(); ++i) {
            const auto& d = detections[i];
            json += "{";
            json += "\"class_id\": " + std::to_string(d.classId) + ", ";
            json += "\"confidence\": " + std::to_string(d.confidence) + ", ";
            json += "\"center\": " + std::to_string(d.center) + ", ";
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

        result_str = json;
        return result_str.c_str();

    } catch (const std::exception& e) {
        result_str = std::string("{\"error\": \"") + e.what() + "\"}";
        return result_str.c_str();
    }
}

void release_detector(void* detector_ptr) {
    if (detector_ptr) {
        BallDetector* detector = static_cast<BallDetector*>(detector_ptr);
        delete detector;
    }
}
}
#endif
