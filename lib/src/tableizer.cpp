#include "tableizer.hpp"

#include <algorithm>
#include <iostream>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <vector>

#include "ball_detection.hpp"
#include "ball_detector.hpp"
#include "ffi_api.hpp"
#include "image_processing.hpp"
#include "json_parser.hpp"
#include "quad_analysis.hpp"
#include "table_detector.hpp"
#include "utilities.hpp"

using namespace std;
using namespace cv;

int runTableizerForImage(Mat image, BallDetector& ballDetector) {
#ifdef PLATFORM_MACOS
    LOGI("--- 1: Table Detection (FFI) ---");
    cv::Mat bgraImage;
    cv::cvtColor(image, bgraImage, cv::COLOR_BGR2BGRA);
    const char* jsonResult =
        detect_table_bgra(bgraImage.data, bgraImage.cols, bgraImage.rows, bgraImage.step, 0);

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
                          bgraImage.step, quadPointsFlat.data(), quadPoints.size(), 0);

    if (ballJsonResult == nullptr) {
        LOGE("Error: Ball detection FFI returned null.");
        return -1;
    }

    // Parse ball detection results
    std::string ballJsonStr(ballJsonResult);
    LOGI("Ball detection result: %s", ballJsonStr.c_str());

    std::vector<Detection> detections = parseDetectionsFromJson(ballJsonStr);
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

    // Get reference dimensions from masked image (for consistent display size)
    int refHeight = bgraImage.rows;
    int refWidth = bgraImage.cols;
    LOGI("Reference dimensions from masked image: %dx%d", refWidth, refHeight);

    // Prepare warpedOut for display (rotate to portrait if needed, resize to match reference)
    cv::Mat warpedDisplay = warpedOut.clone();
    if (warpedDisplay.cols > warpedDisplay.rows) {
        // Landscape -> rotate to portrait
        cv::rotate(warpedDisplay, warpedDisplay, cv::ROTATE_90_CLOCKWISE);
        LOGI("Rotated Warped Table to portrait: %dx%d", warpedDisplay.cols, warpedDisplay.rows);
    }
    // Resize to approximately match reference height
    double scale = (double)refHeight / warpedDisplay.rows;
    cv::resize(warpedDisplay, warpedDisplay, cv::Size(), scale, scale, cv::INTER_LINEAR);

    LOGI("Image size before imshow (Warped Table + Balls): %dx%d", warpedDisplay.cols, warpedDisplay.rows);
    cv::imshow("Warped Table + Balls", warpedDisplay);

    if (!shotStudio.empty()) {
        cv::Mat shotStudioDisplay = shotStudio.clone();
        if (shotStudioDisplay.cols > shotStudioDisplay.rows) {
            // Landscape -> rotate to portrait
            cv::rotate(shotStudioDisplay, shotStudioDisplay, cv::ROTATE_90_CLOCKWISE);
            LOGI("Rotated Shot-Studio to portrait: %dx%d", shotStudioDisplay.cols, shotStudioDisplay.rows);
        }
        // Resize to approximately match reference height
        double scaleStudio = (double)refHeight / shotStudioDisplay.rows;
        cv::resize(shotStudioDisplay, shotStudioDisplay, cv::Size(), scaleStudio, scaleStudio, cv::INTER_LINEAR);

        LOGI("Image size before imshow (Shot-Studio Overlay): %dx%d", shotStudioDisplay.cols,
             shotStudioDisplay.rows);
        cv::imshow("Shot-Studio Overlay", shotStudioDisplay);
    }
    cv::waitKey(0);
#endif

    return 0;
}
