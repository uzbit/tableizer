#include "table_detector.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

#include "utilities.hpp"

CellularTableDetector::CellularTableDetector(int resizeHeight, int cellSize, double deltaEThreshold)
    : resizeHeight(resizeHeight), cellSize(cellSize), deltaEThreshold(deltaEThreshold) {}

Mat CellularTableDetector::prepareImage(const Mat &imgBgr) {
    if (imgBgr.empty()) {
        std::cerr << "Error: input image is empty!" << std::endl;
        return imgBgr;
    }
    Mat small;
    float scale = (float)resizeHeight / imgBgr.rows;
    resize(imgBgr, small, Size(0, 0), scale, scale, INTER_AREA);

    Mat lab;
    cvtColor(small, lab, COLOR_BGR2Lab);

    Mat lab_float;
    lab.convertTo(lab_float, CV_32F);

    vector<Mat> channels(3);
    split(lab_float, channels);
    channels[0] = channels[0] * 100.0 / 255.0;  // L channel
    channels[1] = channels[1] - 128.0;          // a channel
    channels[2] = channels[2] - 128.0;          // b channel
    merge(channels, lab_float);

    return lab_float;
}

Vec3f CellularTableDetector::getMedianLab(const Mat &labImg, const Rect& cellRect) {
    Mat cell = labImg(cellRect);
    Mat cellReshaped = cell.reshape(3, static_cast<int>(cell.total()));
    CV_Assert(cellReshaped.type() == CV_32FC3);

    vector<float> L, A, B;
    L.reserve(cellReshaped.rows);
    A.reserve(cellReshaped.rows);
    B.reserve(cellReshaped.rows);

    for (int i = 0; i < cellReshaped.rows; ++i) {
        const Vec3f v = cellReshaped.at<Vec3f>(i, 0);
        L.push_back(v[0]);
        A.push_back(v[1]);
        B.push_back(v[2]);
    }

    sort(L.begin(), L.end());
    sort(A.begin(), A.end());
    sort(B.begin(), B.end());

    float medianL = L.empty() ? 0 : L[L.size() / 2];
    float medianA = A.empty() ? 0 : A[A.size() / 2];
    float medianB = B.empty() ? 0 : B[B.size() / 2];

    return Vec3f(medianL, medianA, medianB);
}

void CellularTableDetector::detect(const Mat &imgBgr, Mat &mask, Mat &debugDraw) {
    Mat small_bgr;
    float scale = (float)resizeHeight / imgBgr.rows;
    resize(imgBgr, small_bgr, Size(0, 0), scale, scale, INTER_AREA);
    debugDraw = small_bgr.clone();

    Mat labImg = prepareImage(imgBgr);

    int rows = static_cast<int>(ceil((double)labImg.rows / cellSize));
    int cols = static_cast<int>(ceil((double)labImg.cols / cellSize));

    Mat visited = Mat::zeros(rows, cols, CV_8U);
    Mat inside = Mat::zeros(rows, cols, CV_8U);

    int centreR = rows / 2;
    int centreC = cols / 2;

    int y1 = centreR * cellSize;
    int x1 = centreC * cellSize;
    int y2 = min(y1 + cellSize, labImg.rows);
    int x2 = min(x1 + cellSize, labImg.cols);
    Rect centerRect(x1, y1, x2 - x1, y2 - y1);
    Vec3f refLab = getMedianLab(labImg, centerRect);

    vector<Point> queue;
    queue.push_back(Point(centreC, centreR));
    visited.at<uchar>(centreR, centreC) = 1;

    while (!queue.empty()) {
        Point p = queue.back();
        queue.pop_back();
        int r = p.y;
        int c = p.x;

        y1 = r * cellSize;
        x1 = c * cellSize;
        y2 = min(y1 + cellSize, labImg.rows);
        x2 = min(x1 + cellSize, labImg.cols);
        Rect cellRect(x1, y1, x2 - x1, y2 - y1);
        Vec3f cellLab = getMedianLab(labImg, cellRect);
        if (deltaE2000(refLab, cellLab) < deltaEThreshold) {
            inside.at<uchar>(r, c) = 1;
            for (int dr = -1; dr <= 1; ++dr) {
                for (int dc = -1; dc <= 1; ++dc) {
                    if (dr == 0 && dc == 0) continue;
                    int nr = r + dr;
                    int nc = c + dc;
                    if (nr >= 0 && nr < rows && nc >= 0 && nc < cols &&
                        !visited.at<uchar>(nr, nc)) {
                        visited.at<uchar>(nr, nc) = 1;
                        queue.push_back(Point(nc, nr));
                    }
                }
            }
        }
    }

    resize(inside, mask, small_bgr.size(), 0, 0, INTER_NEAREST);
    drawCells(debugDraw, inside);
}

void CellularTableDetector::drawCells(Mat &canvas, const Mat &insideMask) {
    for (int r = 0; r < insideMask.rows; ++r) {
        for (int c = 0; c < insideMask.cols; ++c) {
            if (insideMask.at<uchar>(r, c)) {
                int y = r * cellSize;
                int x = c * cellSize;
                rectangle(canvas, Point(x, y), Point(x + cellSize - 1, y + cellSize - 1),
                          Scalar(0, 255, 0), 1);
            }
        }
    }
}

double CellularTableDetector::deltaE2000(const Vec3f &lab1, const Vec3f &lab2) {
    double L1 = lab1[0], a1 = lab1[1], b1 = lab1[2];
    double L2 = lab2[0], a2 = lab2[1], b2 = lab2[2];

    double c1 = sqrt(a1 * a1 + b1 * b1);
    double c2 = sqrt(a2 * a2 + b2 * b2);
    double cBar = (c1 + c2) / 2.0;

    double G = 0.5 * (1 - sqrt(pow(cBar, 7) / (pow(cBar, 7) + pow(25.0, 7))));
    double a1p = (1 + G) * a1;
    double a2p = (1 + G) * a2;
    double c1p = sqrt(a1p * a1p + b1 * b1);
    double c2p = sqrt(a2p * a2p + b2 * b2);
    double cBarP = (c1p + c2p) / 2.0;

    double h1p = atan2(b1, a1p) * 180.0 / M_PI;
    if (h1p < 0) h1p += 360;
    double h2p = atan2(b2, a2p) * 180.0 / M_PI;
    if (h2p < 0) h2p += 360;

    double dLp = L2 - L1;
    double dCp = c2p - c1p;
    double dhp = h2p - h1p;
    if (c1p * c2p != 0) {
        if (abs(dhp) > 180) {
            dhp -= 360 * (dhp > 0 ? 1 : -1);
        }
    } else {
        dhp = 0;
    }

    double dHp = 2 * sqrt(c1p * c2p) * sin(dhp * M_PI / 180.0 / 2.0);

    double LBarP = (L1 + L2) / 2.0;
    double hBarP = h1p + dhp / 2.0;
    if (c1p * c2p != 0 && abs(h1p - h2p) > 180) {
        hBarP = (h1p + h2p + 360) / 2.0;
    }
    if (hBarP >= 360) hBarP -= 360;

    double T = 1 - 0.17 * cos((hBarP - 30) * M_PI / 180.0) +
               0.24 * cos((2 * hBarP) * M_PI / 180.0) + 0.32 * cos((3 * hBarP + 6) * M_PI / 180.0) -
               0.20 * cos((4 * hBarP - 63) * M_PI / 180.0);

    double Sl = 1 + (0.015 * pow(LBarP - 50, 2)) / sqrt(20 + pow(LBarP - 50, 2));
    double Sc = 1 + 0.045 * cBarP;
    double Sh = 1 + 0.015 * cBarP * T;
    double Rt = -2 * sqrt(pow(cBarP, 7) / (pow(cBarP, 7) + pow(25.0, 7))) *
                sin(60 * exp(-pow((hBarP - 275) / 25.0, 2)) * M_PI / 180.0);

    return sqrt(pow(dLp / Sl, 2) + pow(dCp / Sc, 2) + pow(dHp / Sh, 2) +
                Rt * (dCp / Sc) * (dHp / Sh));
}

vector<Point2f> CellularTableDetector::quadFromInside(const Mat &inside, int width, int height) {
    // Find contours of the inside cells
    vector<vector<Point>> contours;
    findContours(inside, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    if (contours.empty()) {
        return Mat();
    }

    // Find the largest contour
    double maxArea = 0;
    int maxAreaIdx = -1;
    for (size_t i = 0; i < contours.size(); i++) {
        double area = contourArea(contours[i]);
        if (area > maxArea) {
            maxArea = area;
            maxAreaIdx = i;
        }
    }

    if (maxAreaIdx == -1) {
        return Mat();
    }

    // Get the convex hull of the largest contour
    vector<Point> hull;
    convexHull(contours[maxAreaIdx], hull);

    // Approximate the hull with a quadrilateral
    vector<Point> approx;
    approxPolyN(hull, approx, 4, arcLength(hull, true) * 0.2, true);

    if (approx.size() != 4) {
        // Fallback or error handling if not a quadrilateral
        return Mat();
    }

    // Convert to Mat
    Mat quad(4, 1, CV_32FC2);
    for (int i = 0; i < 4; ++i) {
        quad.at<Vec2f>(i, 0) = Point2f(approx[i].x, approx[i].y);
    }

    return orderQuad(quad);
}