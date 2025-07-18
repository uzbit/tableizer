#include "table_detector.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

#include "utilities.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

CellularTableDetector::CellularTableDetector(int resizeHeight, int cellSize, double deltaEThreshold)
    : resizeHeight_(resizeHeight), cellSize_(cellSize), deltaEThreshold_(deltaEThreshold) {}

cv::Mat CellularTableDetector::prepareImage(const cv::Mat &imgBgr) {
    cv::Mat small;
    float scale = (float)resizeHeight_ / imgBgr.rows;
    cv::resize(imgBgr, small, cv::Size(0, 0), scale, scale, cv::INTER_AREA);

    cv::Mat lab;
    cv::cvtColor(small, lab, cv::COLOR_BGR2Lab);

    cv::Mat lab_float;
    lab.convertTo(lab_float, CV_32F);

    std::vector<cv::Mat> channels(3);
    cv::split(lab_float, channels);
    channels[0] = channels[0] * 100.0 / 255.0;  // L channel
    channels[1] = channels[1] - 128.0;          // a channel
    channels[2] = channels[2] - 128.0;          // b channel
    cv::merge(channels, lab_float);

    return lab_float;
}

cv::Vec3f CellularTableDetector::getMedianLab(const cv::Mat &labImg, int cellR, int cellC) {
    int y1 = cellR * cellSize_;
    int x1 = cellC * cellSize_;
    int y2 = std::min(y1 + cellSize_, labImg.rows);
    int x2 = std::min(x1 + cellSize_, labImg.cols);

    cv::Mat cell = labImg(cv::Rect(x1, y1, x2 - x1, y2 - y1)).clone();
    cv::Mat cell_reshaped = cell.reshape(1, cell.total());

    std::vector<float> L, a, b;
    for (int i = 0; i < cell_reshaped.rows; ++i) {
        L.push_back(cell_reshaped.at<cv::Vec3f>(i)[0]);
        a.push_back(cell_reshaped.at<cv::Vec3f>(i)[1]);
        b.push_back(cell_reshaped.at<cv::Vec3f>(i)[2]);
    }

    std::sort(L.begin(), L.end());
    std::sort(a.begin(), a.end());
    std::sort(b.begin(), b.end());

    float medianL = L.empty() ? 0 : L[L.size() / 2];
    float medianA = a.empty() ? 0 : a[a.size() / 2];
    float medianB = b.empty() ? 0 : b[b.size() / 2];

    return cv::Vec3f(medianL, medianA, medianB);
}

void CellularTableDetector::detect(const cv::Mat &imgBgr, cv::Mat &mask, cv::Mat &debugDraw) {
    cv::Mat small_bgr;
    float scale = (float)resizeHeight_ / imgBgr.rows;
    cv::resize(imgBgr, small_bgr, cv::Size(0, 0), scale, scale, cv::INTER_AREA);
    debugDraw = small_bgr.clone();

    cv::Mat labImg = prepareImage(imgBgr);

    int rows = static_cast<int>(std::ceil((double)labImg.rows / cellSize_));
    int cols = static_cast<int>(std::ceil((double)labImg.cols / cellSize_));

    cv::Mat visited = cv::Mat::zeros(rows, cols, CV_8U);
    cv::Mat inside = cv::Mat::zeros(rows, cols, CV_8U);

    int centreR = rows / 2;
    int centreC = cols / 2;

    cv::Vec3f refLab = getMedianLab(labImg, centreR, centreC);

    std::vector<cv::Point> queue;
    queue.push_back(cv::Point(centreC, centreR));
    visited.at<uchar>(centreR, centreC) = 1;

    while (!queue.empty()) {
        cv::Point p = queue.back();
        queue.pop_back();
        int r = p.y;
        int c = p.x;

        cv::Vec3f cellLab = getMedianLab(labImg, r, c);
        if (deltaE2000(refLab, cellLab) < deltaEThreshold_) {
            inside.at<uchar>(r, c) = 1;
            for (int dr = -1; dr <= 1; ++dr) {
                for (int dc = -1; dc <= 1; ++dc) {
                    if (dr == 0 && dc == 0) continue;
                    int nr = r + dr;
                    int nc = c + dc;
                    if (nr >= 0 && nr < rows && nc >= 0 && nc < cols &&
                        !visited.at<uchar>(nr, nc)) {
                        visited.at<uchar>(nr, nc) = 1;
                        queue.push_back(cv::Point(nc, nr));
                    }
                }
            }
        }
    }

    cv::resize(inside, mask, small_bgr.size(), 0, 0, cv::INTER_NEAREST);
    drawCells(debugDraw, inside);
}

void CellularTableDetector::drawCells(cv::Mat &canvas, const cv::Mat &insideMask) {
    for (int r = 0; r < insideMask.rows; ++r) {
        for (int c = 0; c < insideMask.cols; ++c) {
            if (insideMask.at<uchar>(r, c)) {
                int y = r * cellSize_;
                int x = c * cellSize_;
                cv::rectangle(canvas, cv::Point(x, y),
                              cv::Point(x + cellSize_ - 1, y + cellSize_ - 1),
                              cv::Scalar(0, 255, 0), 1);
            }
        }
    }
}

double CellularTableDetector::deltaE2000(const cv::Vec3f &lab1, const cv::Vec3f &lab2) {
    double L1 = lab1[0], a1 = lab1[1], b1 = lab1[2];
    double L2 = lab2[0], a2 = lab2[1], b2 = lab2[2];

    double c1 = std::sqrt(a1 * a1 + b1 * b1);
    double c2 = std::sqrt(a2 * a2 + b2 * b2);
    double cBar = (c1 + c2) / 2.0;

    double G = 0.5 * (1 - std::sqrt(std::pow(cBar, 7) / (std::pow(cBar, 7) + std::pow(25.0, 7))));
    double a1p = (1 + G) * a1;
    double a2p = (1 + G) * a2;
    double c1p = std::sqrt(a1p * a1p + b1 * b1);
    double c2p = std::sqrt(a2p * a2p + b2 * b2);
    double cBarP = (c1p + c2p) / 2.0;

    double h1p = std::atan2(b1, a1p) * 180.0 / M_PI;
    if (h1p < 0) h1p += 360;
    double h2p = std::atan2(b2, a2p) * 180.0 / M_PI;
    if (h2p < 0) h2p += 360;

    double dLp = L2 - L1;
    double dCp = c2p - c1p;
    double dhp = h2p - h1p;
    if (c1p * c2p != 0) {
        if (std::abs(dhp) > 180) {
            dhp -= 360 * (dhp > 0 ? 1 : -1);
        }
    } else {
        dhp = 0;
    }

    double dHp = 2 * std::sqrt(c1p * c2p) * std::sin(dhp * M_PI / 180.0 / 2.0);

    double LBarP = (L1 + L2) / 2.0;
    double hBarP = h1p + dhp / 2.0;
    if (c1p * c2p != 0 && std::abs(h1p - h2p) > 180) {
        hBarP = (h1p + h2p + 360) / 2.0;
    }
    if (hBarP >= 360) hBarP -= 360;

    double T = 1 - 0.17 * std::cos((hBarP - 30) * M_PI / 180.0) +
               0.24 * std::cos((2 * hBarP) * M_PI / 180.0) +
               0.32 * std::cos((3 * hBarP + 6) * M_PI / 180.0) -
               0.20 * std::cos((4 * hBarP - 63) * M_PI / 180.0);

    double Sl = 1 + (0.015 * std::pow(LBarP - 50, 2)) / std::sqrt(20 + std::pow(LBarP - 50, 2));
    double Sc = 1 + 0.045 * cBarP;
    double Sh = 1 + 0.015 * cBarP * T;
    double Rt = -2 * std::sqrt(std::pow(cBarP, 7) / (std::pow(cBarP, 7) + std::pow(25.0, 7))) *
                std::sin(60 * std::exp(-std::pow((hBarP - 275) / 25.0, 2)) * M_PI / 180.0);

    return std::sqrt(std::pow(dLp / Sl, 2) + std::pow(dCp / Sc, 2) + std::pow(dHp / Sh, 2) +
                     Rt * (dCp / Sc) * (dHp / Sh));
}

std::vector<cv::Point2f> CellularTableDetector::quadFromInside(const cv::Mat &inside, int width,
                                                               int height) {
    // Find contours of the inside cells
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(inside, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    if (contours.empty()) {
        return cv::Mat();
    }

    // Find the largest contour
    double maxArea = 0;
    int maxAreaIdx = -1;
    for (size_t i = 0; i < contours.size(); i++) {
        double area = cv::contourArea(contours[i]);
        if (area > maxArea) {
            maxArea = area;
            maxAreaIdx = i;
        }
    }

    if (maxAreaIdx == -1) {
        return cv::Mat();
    }

    // Get the convex hull of the largest contour
    std::vector<cv::Point> hull;
    cv::convexHull(contours[maxAreaIdx], hull);

    // Approximate the hull with a quadrilateral
    std::vector<cv::Point> approx;
    cv::approxPolyN(hull, approx, 4, cv::arcLength(hull, true) * 0.2, true);

    if (approx.size() != 4) {
        // Fallback or error handling if not a quadrilateral
        return cv::Mat();
    }

    // Convert to Mat
    cv::Mat quad(4, 1, CV_32FC2);
    for (int i = 0; i < 4; ++i) {
        quad.at<cv::Vec2f>(i, 0) = cv::Point2f(approx[i].x, approx[i].y);
    }

    return orderQuad(quad);
}