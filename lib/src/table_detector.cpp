#include "table_detector.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include "utilities.hpp"

CellularTableDetector::CellularTableDetector(int resizeHeight, int cellSize, double deltaEThreshold)
    : resizeHeight(resizeHeight), cellSize(cellSize), deltaEThreshold(deltaEThreshold) {}

Vec3f CellularTableDetector::getMedianLab(const Mat &labImg, const Rect &cellRect) {
    // Use systematic grid sampling for more consistent results
    const int SAMPLE_STRIDE = 4;
    Mat cell = labImg(cellRect);

    if (cell.rows == 0 || cell.cols == 0) {
        return Vec3f(0, 0, 0);
    }

    vector<float> L, A, B;

    // Use systematic grid sampling instead of random sampling
    int stepR = SAMPLE_STRIDE;
    int stepC = SAMPLE_STRIDE;
    for (int r = 0; r < cell.rows; r += stepR) {
        for (int c = 0; c < cell.cols; c += stepC) {
            const Vec3f &pixel = cell.at<Vec3f>(r, c);
            L.push_back(pixel[0]);
            A.push_back(pixel[1]);
            B.push_back(pixel[2]);
        }
    }

    if (L.empty()) {
        // Fallback to center pixel if grid sampling failed
        const Vec3f &centerPixel = cell.at<Vec3f>(cell.rows / 2, cell.cols / 2);
        return centerPixel;
    }

    // Find median of the systematic samples
    size_t mid = L.size() / 2;
    nth_element(L.begin(), L.begin() + mid, L.end());
    nth_element(A.begin(), A.begin() + mid, A.end());
    nth_element(B.begin(), B.begin() + mid, B.end());

    return Vec3f(L[mid], A[mid], B[mid]);
}

void CellularTableDetector::precomputeLabCache(const Mat &labImg, int rows, int cols) {
    // Initialize cache with proper dimensions
    labCache.resize(rows);
    for (int r = 0; r < rows; ++r) {
        labCache[r].resize(cols);
    }

    // Precompute LAB values for all cells
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            int y1 = r * cellSize;
            int x1 = c * cellSize;
            int y2 = min(y1 + cellSize, labImg.rows);
            int x2 = min(x1 + cellSize, labImg.cols);
            Rect cellRect(x1, y1, x2 - x1, y2 - y1);

            // Cache the median LAB value for this cell
            labCache[r][c] = getMedianLab(labImg, cellRect);
        }
    }
}

void CellularTableDetector::detect(const Mat &imgBgra, Mat &mask, Mat &debugDraw,
                                   int rotationDegrees) {
    // --- Image Rotation ---
    cv::Mat rotated_bgra;
    if (rotationDegrees == 90) {
        cv::rotate(imgBgra, rotated_bgra, cv::ROTATE_90_CLOCKWISE);
    } else if (rotationDegrees == 270) {
        cv::rotate(imgBgra, rotated_bgra, cv::ROTATE_90_COUNTERCLOCKWISE);
    } else if (rotationDegrees == 180) {
        cv::rotate(imgBgra, rotated_bgra, cv::ROTATE_180);
    } else {
        rotated_bgra = imgBgra;
    }

    Mat small_bgr;
    float scale = (float)resizeHeight / rotated_bgra.rows;
    resize(rotated_bgra, small_bgr, Size(0, 0), scale, scale, INTER_LINEAR);
    debugDraw = small_bgr.clone();

    // --- Single, Upfront LAB Conversion ---
    Mat lab_image;
    cvtColor(small_bgr, lab_image, COLOR_BGR2Lab);
    Mat lab_float;
    vector<Mat> lab_channels_8u;
    split(lab_image, lab_channels_8u);

    Mat l_float, a_float, b_float;
    lab_channels_8u[0].convertTo(l_float, CV_32F, 100.0 / 255.0);
    lab_channels_8u[1].convertTo(a_float, CV_32F, 1.0, -128.0);
    lab_channels_8u[2].convertTo(b_float, CV_32F, 1.0, -128.0);

    vector<Mat> float_channels = {l_float, a_float, b_float};
    merge(float_channels, lab_float);
    // --- End Conversion ---

    int rows = static_cast<int>(ceil((double)lab_float.rows / cellSize));
    int cols = static_cast<int>(ceil((double)lab_float.cols / cellSize));

    // Precompute all LAB values once
    precomputeLabCache(lab_float, rows, cols);

    Mat visited = Mat::zeros(rows, cols, CV_8U);
    Mat inside = Mat::zeros(rows, cols, CV_8U);

    int centreR = rows / 2;
    int centreC = cols / 2;

    // Calculate robust multi-reference color instead of single center point
    Vec3f refLab = calculateMultiReferenceColor(lab_float, rows, cols);
    cout << "Multi-reference color: L=" << refLab[0] << ", A=" << refLab[1] << ", B=" << refLab[2]
         << endl;

    // Calculate adaptive threshold based on local color variance
    double adaptiveThresh = calculateAdaptiveThreshold(lab_float, refLab);
    cout << "Adaptive threshold: " << adaptiveThresh << " (initial: " << deltaEThreshold << ")"
         << endl;

    vector<Point> queue;
    queue.push_back(Point(centreC, centreR));
    visited.at<uchar>(centreR, centreC) = 1;

    while (!queue.empty()) {
        Point p = queue.back();
        queue.pop_back();
        int r = p.y;
        int c = p.x;

        // Get LAB value from cache instead of recalculating
        Vec3f cellLab = labCache[r][c];
        if (colorDelta(refLab, cellLab) < adaptiveThresh) {
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

double CellularTableDetector::colorDelta(const Vec3f &lab1, const Vec3f &lab2) {
    double dL = lab1[0] - lab2[0];
    double da = lab1[1] - lab2[1];
    double db = lab1[2] - lab2[2];
    return sqrt(dL * dL + da * da + db * db);
}

vector<Point2f> CellularTableDetector::getQuadFromMask(const Mat &inside) {
    // Find contours of the inside cells
    vector<vector<Point>> contours;
    findContours(inside, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    if (contours.empty()) {
        return vector<Point2f>();
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
        return vector<Point2f>();
    }

    // Get the convex hull of the largest contour
    vector<Point> hull;
    convexHull(contours[maxAreaIdx], hull);

    // Approximate the hull with a quadrilateral
    vector<Point> approx;
    approxPolyN(hull, approx, 4, arcLength(hull, true) * 0.2, true);

    if (approx.size() != 4) {
        // Fallback or error handling if not a quadrilateral
        return vector<Point2f>();
    }

    // Convert to vector<Point2f>
    vector<Point2f> quadPoints;
    quadPoints.reserve(4);
    for (int i = 0; i < 4; ++i) {
        quadPoints.emplace_back(static_cast<float>(approx[i].x), static_cast<float>(approx[i].y));
    }

    return orderQuad(quadPoints);
}

double CellularTableDetector::calculateAdaptiveThreshold(const Mat &labImg, const Vec3f &refLab) {
    // Calculate a sophisticated adaptive threshold based on local color variance
    // This analyzes the color distribution around the reference point to determine
    // how strict the threshold should be

    int rows = labCache.size();
    int cols = rows > 0 ? labCache[0].size() : 0;

    if (rows == 0 || cols == 0) return deltaEThreshold;

    int centerR = rows / 2;
    int centerC = cols / 2;

    // Sample colors only from a reasonable table area around center (not entire image)
    // This avoids including background colors that aren't relevant for table detection
    vector<double> deltaEs;
    deltaEs.reserve(100);

    // Define a reasonable sampling area around center (limit to ~1/SAMPLING_AREA_DIVISOR of image size)
    int maxRadius = std::min({rows / 6, cols / 6, 15});  // Reasonable table area, not entire image

    // Sample in a grid pattern within the table area instead of rings
    for (int dr = -maxRadius; dr <= maxRadius; dr += 2) {  // Step by SAMPLING_STEP for efficiency
        for (int dc = -maxRadius; dc <= maxRadius; dc += 2) {
            int sampleR = centerR + dr;
            int sampleC = centerC + dc;

            // Check bounds and skip center point
            if (sampleR >= 0 && sampleR < rows && sampleC >= 0 && sampleC < cols &&
                !(dr == 0 && dc == 0)) {
                Vec3f sampleLab = labCache[sampleR][sampleC];
                double deltaE = colorDelta(refLab, sampleLab);
                deltaEs.push_back(deltaE);
            }
        }
    }

    if (deltaEs.empty()) {
        return deltaEThreshold;  // Fallback to original threshold
    }

    // Calculate sophisticated statistics
    sort(deltaEs.begin(), deltaEs.end());

    size_t n = deltaEs.size();
    double q25 = deltaEs[n / 4];      // Q1_PERCENTILE percentile
    double median = deltaEs[n / 2];   // MEDIAN_PERCENTILE percentile
    double q75 = deltaEs[3 * n / 4];  // Q3_PERCENTILE percentile
    double iqr = q75 - q25;           // Interquartile range

    // Calculate mean and standard deviation
    double mean = 0.0;
    for (double de : deltaEs) mean += de;
    mean /= n;

    double variance = 0.0;
    for (double de : deltaEs) {
        double diff = de - mean;
        variance += diff * diff;
    }
    double stdDev = sqrt(variance / n);

    // Calculate adaptive threshold based on the data distribution
    // For uniform table surfaces, we want to be more permissive
    double adaptiveThresh;

    if (stdDev < 2.0) {
        // Very uniform area - be quite permissive
        adaptiveThresh = std::max(deltaEThreshold * 0.8, q75 + iqr * 2.0);
    } else if (stdDev < 5.0) {
        // Moderately uniform - still be permissive
        adaptiveThresh = std::max(deltaEThreshold * 0.7, q75 + iqr);
    } else {
        // High variance area - use more conservative approach
        adaptiveThresh = std::max(deltaEThreshold * 0.6, median + iqr);
    }

    // Cap at reasonable maximum but don't force minimum bounds
    adaptiveThresh = std::min(deltaEThreshold * 1.5, adaptiveThresh);

    cout << "Adaptive threshold analysis:" << endl;
    cout << "  Samples: " << n << ", Mean: " << mean << ", StdDev: " << stdDev << endl;
    cout << "  Q25: " << q25 << ", Median: " << median << ", Q75: " << q75 << ", IQR: " << iqr
         << endl;
    cout << "  Final adaptive threshold: " << adaptiveThresh << endl;

    return adaptiveThresh;
}

Vec3f CellularTableDetector::calculateMultiReferenceColor(const Mat &labImg, int rows, int cols) {
    // Calculate a robust multi-reference color using systematic sampling
    // This provides much more stable color reference than a single center point

    if (rows == 0 || cols == 0) return Vec3f(50, 0, 0);  // Default neutral

    int centerR = rows / 2;
    int centerC = cols / 2;

    vector<Vec3f> candidateColors;
    candidateColors.reserve(25);  // Up to GRID_SIZE x GRID_SIZE grid

    // Sample in a systematic GRID_SIZE x GRID_SIZE grid around center (adaptive to image size)
    int maxOffset = std::min({3, rows / 4, cols / 4});  // Limit offset based on image size

    for (int dr = -maxOffset; dr <= maxOffset; dr++) {
        for (int dc = -maxOffset; dc <= maxOffset; dc++) {
            int sampleR = centerR + dr;
            int sampleC = centerC + dc;

            // Check bounds
            if (sampleR >= 0 && sampleR < rows && sampleC >= 0 && sampleC < cols) {
                Vec3f sampleColor = labCache[sampleR][sampleC];
                candidateColors.push_back(sampleColor);
            }
        }
    }

    if (candidateColors.empty()) {
        // Fallback to center if no samples
        return labCache[centerR][centerC];
    }

    // Calculate median reference color directly (median handles outliers naturally)
    return calculateMedianColor(candidateColors);
}

Vec3f CellularTableDetector::calculateMedianColor(const vector<Vec3f> &colors) {
    if (colors.empty()) {
        return Vec3f(50, 0, 0);  // Default neutral
    }

    if (colors.size() == 1) {
        return colors[0];
    }

    // Calculate median for each channel independently
    vector<float> L, A, B;
    L.reserve(colors.size());
    A.reserve(colors.size());
    B.reserve(colors.size());

    for (const auto &color : colors) {
        L.push_back(color[0]);
        A.push_back(color[1]);
        B.push_back(color[2]);
    }

    // Find medians
    size_t mid = colors.size() / 2;
    nth_element(L.begin(), L.begin() + mid, L.end());
    nth_element(A.begin(), A.begin() + mid, A.end());
    nth_element(B.begin(), B.begin() + mid, B.end());

    return Vec3f(L[mid], A[mid], B[mid]);
}
