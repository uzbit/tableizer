
#include <algorithm>
#include <iostream>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <vector>

#include "ball_detector.hpp"
#include "table_detector.hpp"
#include "utilities.hpp"

using namespace std;
using namespace cv;

#define IMSHOW true

int runTableizerForImage(Mat image, BallDetector ballDetector) {
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

    for (const auto &pt : quadPoints) quadDraw.emplace_back(cvRound(pt.x), cvRound(pt.y));

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
    for (const auto &p : quadPoints) {
        cout << "  - (" << p.x << ", " << p.y << ")" << endl;
    }
    cout << endl;
#endif

    // 3. Warp Table
    bool rotate = false;
    WarpResult warpResult = warpTable(image, quadPoints, "warp.jpg", 840, rotate);

    // --- Ball detection & drawing --------------------------
    // 4. Detect balls **on the original image**
    cout << "--- Step 3: Ball Detection ---" << endl;
    const vector<Detection> detections = ballDetector.detect(image, 0.6, 0.5);
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
    const int radius = 14;
    const cv::Scalar textColor(255, 255, 255);  // white id

    if (detections.empty()) {
        cout << "No balls detected.\n\n";
    } else {
        // centres in *original* pixel space
        vector<cv::Point2f> ballCentresOrig;
        for (const auto &d : detections) {
            cv::Point2f p(d.box.x + d.box.width * 0.5f, d.box.y + d.box.height * 0.5f);
            ballCentresOrig.emplace_back(p);
            cout << "Ball at @ " << p << "\n";
        }

        // map straight to canonical table space
        vector<cv::Point2f> ballCentresCanonical;
        cv::perspectiveTransform(ballCentresOrig, ballCentresCanonical, Htotal);

        cout << "--- Step 4: Final Ball Locations ---\n";
        float textSize = 3.0;
        for (size_t i = 0; i < ballCentresCanonical.size(); ++i) {
            const auto &p = ballCentresCanonical[i];
            cout << "  • class " << detections[i].classId << " @ (" << p.x << ", " << p.y << ")\n";
#if IMSHOW
            cv::Scalar ballColor;
            if (detections[i].classId == 3)
                ballColor = cv::Scalar(0, 0, 255);  // red fill
            else if (detections[i].classId == 2)
                ballColor = cv::Scalar(255, 222, 33);  // yellow fill
            else if (detections[i].classId == 1)
                ballColor = cv::Scalar(255, 255, 255);  // cue fill
            else
                ballColor = cv::Scalar(0, 0, 0);  // black fill

            cv::circle(warpedOut, p, 5, ballColor, cv::LineTypes::LINE_AA);
            cv::putText(warpedOut, std::to_string(detections[i].classId),
                        p + cv::Point2f(radius + 2, 0), cv::FONT_HERSHEY_SIMPLEX, textSize,
                        textColor, 2);

            // draw on studio template (assumes same canvas size)
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