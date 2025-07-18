#ifndef TABLE_DETECTOR_HPP
#define TABLE_DETECTOR_HPP

#include <opencv2/opencv.hpp>

class CellularTableDetector {
   public:
    CellularTableDetector(int resizeHeight = 600, int cellSize = 24, double deltaEThreshold = 10.0);

    void detect(const cv::Mat &imgBgr, cv::Mat &mask, cv::Mat &debugDraw);
    cv::Mat quadFromInside(const cv::Mat &inside, int width, int height);

   private:
    cv::Mat prepareImage(const cv::Mat &imgBgr);
    cv::Vec3f getMedianLab(const cv::Mat &labImg, int cellR, int cellC);
    double deltaE2000(const cv::Vec3f &lab1, const cv::Vec3f &lab2);
    void drawCells(cv::Mat &canvas, const cv::Mat &insideMask);

    int resizeHeight_;
    int cellSize_;
    double deltaEThreshold_;
};

#endif  // TABLE_DETECTOR_HPP
