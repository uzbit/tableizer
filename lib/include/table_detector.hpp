#ifndef TABLE_DETECTOR_HPP
#define TABLE_DETECTOR_HPP

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

class CellularTableDetector {
   public:
    CellularTableDetector(int resizeHeight = 600, int cellSize = 24, double deltaEThreshold = 10.0);

    void detect(const Mat &imgBgra, Mat &mask, Mat &debugDraw, int rotationDegrees);
    vector<Point2f> quadFromInside(const Mat &inside, int width, int height);

   private:
    Vec3f getMedianLab(const Mat &bgraCell);
    double deltaE2000(const Vec3f &lab1, const Vec3f &lab2);
    void drawCells(Mat &canvas, const Mat &insideMask);

    int resizeHeight;
    int cellSize;
    double deltaEThreshold;
};

#endif  // TABLE_DETECTOR_HPP
