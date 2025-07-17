#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <numeric>
#include <algorithm>
#include "table_detector.hpp"
#include "ball_detector.hpp"

// Function to order the quad points counter-clockwise
std::vector<cv::Point2f> orderQuad(const std::vector<cv::Point2f>& pts) {
    std::vector<cv::Point2f> sorted_pts = pts;
    cv::Scalar centroid = cv::mean(pts);
    std::sort(sorted_pts.begin(), sorted_pts.end(),
              [centroid](const cv::Point2f& a, const cv::Point2f& b) {
        return std::atan2(a.y - centroid[1], a.x - centroid[0]) <
               std::atan2(b.y - centroid[1], b.x - centroid[0]);
    });
    return sorted_pts;
}

int main() {
    // 1. Load Image
    std::string imagePath = "data/Photos-1-001/P_20250711_201047.jpg";
    cv::Mat image = cv::imread(imagePath);
    if (image.empty()) {
        std::cerr << "Error: Could not open image at " << imagePath << std::endl;
        return -1;
    }
    std::cout << "--- Step 1: Image Loading ---" << std::endl;
    std::cout << "Original image dimensions: " << image.cols << "x" << image.rows << std::endl;
    std::cout << std::endl;

    // 2. Detect Table
    int resizeHeight = 3000;
    int cellSize = 20;
    double deltaEThreshold = 20.0;
    CellularTableDetector table_detector(resizeHeight, cellSize, deltaEThreshold);
    
    std::cout << "--- Step 2: Table Detection ---" << std::endl;
    std::cout << "Parameters: resizeHeight=" << resizeHeight << ", cellSize=" << cellSize << ", deltaEThreshold=" << deltaEThreshold << std::endl;

    cv::Mat mask, resized_for_detection;
    table_detector.detect(image, mask, resized_for_detection);
    std::cout << "Resized image for detection dimensions: " << resized_for_detection.cols << "x" << resized_for_detection.rows << std::endl;

    cv::Mat quad_mat = table_detector.quadFromInside(mask, resized_for_detection.cols, resized_for_detection.rows);

    if (quad_mat.empty()) {
        std::cerr << "Error: Could not detect table quad." << std::endl;
        cv::imshow("Debug: No Quad Found", resized_for_detection);
        cv::waitKey(0);
        return -1;
    }

    std::vector<cv::Point2f> src_quad_unordered;
    for(int i=0; i<4; ++i) {
        src_quad_unordered.push_back(cv::Point2f(quad_mat.at<cv::Vec2f>(i,0)[0], quad_mat.at<cv::Vec2f>(i,0)[1]));
    }
    
    std::vector<cv::Point2f> src_quad = orderQuad(src_quad_unordered);
    std::cout << "Detected table quad corners (in resized image coordinates):" << std::endl;
    for(const auto& p : src_quad) {
        std::cout << "  - (" << p.x << ", " << p.y << ")" << std::endl;
    }
    std::cout << std::endl;

    // 3. Warp Table
    const float WARP_WIDTH = 840.0f;
    const float WARP_HEIGHT = 1680.0f;
    std::cout << "--- Step 3: Warping Table ---" << std::endl;
    std::cout << "Warping to destination size: " << WARP_WIDTH << "x" << WARP_HEIGHT << std::endl;
    
    std::vector<cv::Point2f> dst_quad = {
        {0.0f, 0.0f}, {0.0f, WARP_HEIGHT}, {WARP_WIDTH, WARP_HEIGHT}, {WARP_WIDTH, 0.0f}
    };

    cv::Mat perspective_matrix = cv::getPerspectiveTransform(src_quad, dst_quad);
    cv::Mat warped_table;
    cv::warpPerspective(resized_for_detection, warped_table, perspective_matrix, cv::Size(WARP_WIDTH, WARP_HEIGHT));
    std::cout << std::endl;

    // 4. Detect Balls
    std::cout << "--- Step 4: Ball Detection ---" << std::endl;
    std::string modelPath = "lib/models/detection_model.torchscript.pt";
    BallDetector ball_detector(modelPath);
    std::vector<Detection> detections = ball_detector.detect(image);
    std::cout << "Found " << detections.size() << " balls after non-maximum suppression." << std::endl;
    std::cout << std::endl;

    // 5. Transform Ball Coordinates & Print Final Results
    std::cout << "--- Step 5: Final Ball Locations ---" << std::endl;
    if (!detections.empty()) {
        std::vector<cv::Point2f> ball_centers;
        for (const auto& det : detections) {
            ball_centers.push_back(cv::Point2f(det.box.x + det.box.width / 2.0f, det.box.y + det.box.height / 2.0f));
        }

        float scale = (float)resizeHeight / image.rows;
        std::vector<cv::Point2f> scaled_ball_centers;
        for(const auto& p : ball_centers) {
            scaled_ball_centers.push_back(p * scale);
        }

        std::vector<cv::Point2f> warped_ball_centers;
        cv::perspectiveTransform(scaled_ball_centers, warped_ball_centers, perspective_matrix);

        std::cout << "Predicted ball locations and classes on canonical table:" << std::endl;
        for (size_t i = 0; i < warped_ball_centers.size(); ++i) {
            std::cout << "  - Class: " << detections[i].class_id 
                      << ", Location: (" << warped_ball_centers[i].x 
                      << ", " << warped_ball_centers[i].y << ")" << std::endl;
            
            // 6. Draw Balls on Warped Table
            cv::circle(warped_table, warped_ball_centers[i], 14, cv::Scalar(0, 0, 255), -1);
            cv::putText(warped_table, std::to_string(detections[i].class_id), warped_ball_centers[i] + cv::Point2f(10,0), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255,255,255), 2);
        }
    } else {
        std::cout << "No balls detected." << std::endl;
    }
    std::cout << std::endl;

    // 7. Display Result
    cv::imshow("Final Table State", warped_table);
    cv::waitKey(0);

    return 0;
}