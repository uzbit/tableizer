#ifndef TABLEIZER_HPP
#define TABLEIZER_HPP

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

// A placeholder for the final library API.

struct Ball {
    cv::Point2f position;
    int class_id;
};

class TableState {
   public:
    cv::Mat getWarpedTable() const;
    std::vector<Ball> getBalls() const;

   private:
    cv::Mat warped_table_;
    std::vector<Ball> balls_;
};

TableState detect_table_state(const cv::Mat &image, const std::string &model_path);

#endif  // TABLEIZER_HPP
