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

// Function to order the quad points counter-clockwise
vector<Point2f> orderQuad(const vector<Point2f> &pts) {
    vector<Point2f> sorted_pts = pts;
    Scalar centroid = mean(pts);
    sort(sorted_pts.begin(), sorted_pts.end(), [centroid](const Point2f &a, const Point2f &b) {
        return atan2(a.y - centroid[1], a.x - centroid[0]) <
               atan2(b.y - centroid[1], b.x - centroid[0]);
    });
    return sorted_pts;
}

int main() {
    // 1. Load Image
    string imagePath = "data/Photos-1-001/P_20250711_201047.jpg";
    Mat image = imread(imagePath);
    if (image.empty()) {
        cerr << "Error: Could not open image at " << imagePath << endl;
        return -1;
    }
    cout << "--- Step 1: Image Loading ---" << endl;
    cout << "Original image dimensions: " << image.cols << "x" << image.rows << endl;
    cout << endl;
    imshow("Table", image);
    waitKey(0);
    // 2. Detect Table
    int resizeHeight = 3000;
    int cellSize = 20;
    double deltaEThreshold = 20.0;
    CellularTableDetector table_detector(image.rows, cellSize, deltaEThreshold);

    cout << "--- Step 2: Table Detection ---" << endl;
    cout << "Parameters: resizeHeight=" << resizeHeight << ", cellSize=" << cellSize
         << ", deltaEThreshold=" << deltaEThreshold << endl;

    Mat mask, table_detection;

    table_detector.detect(image, mask, table_detection);
    cout << "Resized image for detection dimensions: " << table_detection.cols << "x"
         << table_detection.rows << endl;

    imshow("Quad Found", table_detection);
    waitKey(0);
    Mat quad_mat = table_detector.quadFromInside(mask, table_detection.cols, table_detection.rows);

    if (quad_mat.empty()) {
        cerr << "Error: Could not detect table quad." << endl;
        imshow("Debug: No Quad Found", table_detection);
        waitKey(0);
        return -1;
    }

    vector<Point2f> src_quad_unordered;
    for (int i = 0; i < 4; ++i) {
        src_quad_unordered.push_back(
            Point2f(quad_mat.at<Vec2f>(i, 0)[0], quad_mat.at<Vec2f>(i, 0)[1]));
    }

    vector<Point2f> src_quad = orderQuad(src_quad_unordered);
    cout << "Detected table quad corners (in resized image coordinates):" << endl;
    for (const auto &p : src_quad) {
        cout << "  - (" << p.x << ", " << p.y << ")" << endl;
    }
    cout << endl;

    // 3. Warp Table
    WarpResult warpResult = warpTable(image, src_quad, "warp.jpg", 840, true);

    // --- Ball detection & drawing --------------------------
    // 4. Detect balls **on the original image**
    cout << "--- Step 4: Ball Detection ---" << endl;
    const string modelPath = "lib/models/detection_model.torchscript.pt";
    BallDetector ballDetector(modelPath);
    const vector<Detection> detections = ballDetector.detect(image);
    cout << "Found " << detections.size() << " balls after non-maximum suppression.\n\n";

    // 5. Build transform: original-pixel  ➜  table_detection  ➜  canonical table
    // --------------------------------------------------------
    // table_detector resized the image to `resizeHeight` while preserving aspect.
    // Derive the scale used for that resize so we can link both spaces together.
    const double scaleY =
        static_cast<double>(table_detection.rows) / static_cast<double>(image.rows);
    const double scaleX =
        static_cast<double>(table_detection.cols) / static_cast<double>(image.cols);

    // homogeneous scale matrix (3×3)
    cv::Mat Hscale = (cv::Mat_<double>(3, 3) << scaleX, 0, 0, 0, scaleY, 0, 0, 0, 1);

    // Make sure both matrices share the same depth
    cv::Mat Hwarp;
    warpResult.transform.convertTo(Hwarp, CV_64F);  // canonical ← resized
    cv::Mat Htotal = Hscale * Hwarp;                // canonical ← original

    // 6. Draw predictions on the canonical table and shot-studio template
    // --------------------------------------------------------
    string studioPath = "data/shotstudio_table_felt_only.png";
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

        cout << "--- Step 5: Final Ball Locations ---\n";
        float textSize = 3.0;
        for (size_t i = 0; i < ballCentresCanonical.size(); ++i) {
            const auto &p = ballCentresCanonical[i];
            cout << "  • class " << detections[i].class_id << " @ (" << p.x << ", " << p.y << ")\n";
            cv::Scalar ballColor;
            if (detections[i].class_id == 0)
                ballColor = cv::Scalar(0, 0, 255);  // red fill
            else if (detections[i].class_id == 1)
                ballColor = cv::Scalar(255, 222, 33);  // yellow fill
            else if (detections[i].class_id == 2)
                ballColor = cv::Scalar(255, 255, 255);  // cue fill
            else
                ballColor = cv::Scalar(0, 0, 0);  // black fill

            cv::putText(warpedOut, std::to_string(detections[i].class_id),
                        p + cv::Point2f(radius + 2, 0), cv::FONT_HERSHEY_SIMPLEX, textSize,
                        textColor, 2);

            // draw on studio template (assumes same canvas size)
            if (!shotStudio.empty()) {
                cv::circle(shotStudio, p, radius, ballColor, cv::FILLED);
                cv::putText(shotStudio, std::to_string(detections[i].class_id),
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
}