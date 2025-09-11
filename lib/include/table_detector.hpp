#ifndef TABLE_DETECTOR_HPP
#define TABLE_DETECTOR_HPP

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

class CellularTableDetector {
   public:
    CellularTableDetector(int resizeHeight, int cellSize, double deltaEThreshold);

    void detect(const Mat &imgBgra, Mat &mask, Mat &debugDraw, int rotationDegrees);
    vector<Point2f> getQuadFromMask(const Mat &inside);
    void drawCells(Mat &canvas, const Mat &insideMask);

   private:
    Vec3f getMedianLab(const Mat &labImg, const Rect &cellRect);
    void precomputeLabCache(const Mat &labImg, int rows, int cols);
    double colorDelta(const Vec3f &lab1, const Vec3f &lab2);
    double calculateAdaptiveThreshold(const Mat &labImg, const Vec3f &refLab);
    Vec3f calculateMultiReferenceColor(const Mat &labImg, int rows, int cols);
    Vec3f calculateMedianColor(const vector<Vec3f> &colors);

    int resizeHeight;
    int cellSize;
    double deltaEThreshold;

    // Cache for precomputed LAB values
    vector<vector<Vec3f>> labCache;
};

#endif  // TABLE_DETECTOR_HPP
