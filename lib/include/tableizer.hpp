#ifndef TABLEIZER_HPP
#define TABLEIZER_HPP

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "ball_detector.hpp"

// A placeholder for the final library API.
using namespace cv;

struct Ball {
    Point2f position;
    int classId;
};

class TableState {
   public:
    Mat getWarpedTable() const;
    std::vector<Ball> getBalls() const;

   private:
    Mat warpedTable;
    std::vector<Ball> balls;
};

int runTableizerForImage(Mat image, BallDetector& ballDetector);

#endif  // TABLEIZER_HPP
