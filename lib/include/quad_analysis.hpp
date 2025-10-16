#ifndef QUAD_ANALYSIS_HPP
#define QUAD_ANALYSIS_HPP

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

using namespace std;
using namespace cv;

enum QuadOrientation { SHORT_SIDE, LONG_SIDE, TOP_DOWN, OTHER };

struct ViewValidation {
    bool isValid;
    bool isLandscape;
    bool isCorrectAspectRatio;
    bool isShortSideView;
    float imageAspectRatio;
    std::string errorMessage;
};

class QuadAnalysis {
   public:
    /**
     * Check if top and bottom lines of quad are parallel within epsilon tolerance
     * @param quad Quadrilateral points (assumed ordered: top-left, top-right, bottom-right,
     * bottom-left)
     * @param epsilon Tolerance for parallel check (default 0.0038)
     * @return true if top and bottom lines are parallel within epsilon
     */
    static bool topBottomParallel(const vector<Point2f>& quad, double epsilon = 0.0038);

    /**
     * Check if left and right lines of quad are parallel within epsilon tolerance
     * @param quad Quadrilateral points (assumed ordered: top-left, top-right, bottom-right,
     * bottom-left)
     * @param epsilon Tolerance for parallel check (default 0.0038)
     * @return true if left and right lines are parallel within epsilon
     */
    static bool leftRightParallel(const vector<Point2f>& quad, double epsilon = 0.0038);

    /**
     * Compute ratio of top line length to right line length
     * @param quad Quadrilateral points (assumed ordered: top-left, top-right, bottom-right,
     * bottom-left)
     * @return Ratio of top line length / right line length
     */
    static double topRightRatio(const vector<Point2f>& quad);

    /**
     * Compute ratio of top line length to left line length
     * @param quad Quadrilateral points (assumed ordered: top-left, top-right, bottom-right,
     * bottom-left)
     * @return Ratio of top line length / left line length
     */
    static double topLeftRatio(const vector<Point2f>& quad);

    /**
     * Determine quad orientation based on parallelism and aspect ratios
     * @param quad Quadrilateral points (assumed ordered: top-left, top-right, bottom-right,
     * bottom-left)
     * @return QuadOrientation enum value
     */
    static QuadOrientation orientation(const vector<Point2f>& quad);

    /**
     * Convert QuadOrientation enum to string
     * @param orientation The orientation enum value
     * @return String representation of orientation
     */
    static string orientationToString(QuadOrientation orientation);

    /**
     * Validate if image and quad meet landscape 16:9 short-side view requirements
     * @param quad Quadrilateral points (assumed ordered: top-left, top-right, bottom-right,
     * bottom-left)
     * @param imageSize Size of the input image
     * @param orientation The determined quad orientation
     * @return ViewValidation struct with validation results and error messages
     */
    static ViewValidation validateLandscapeShortSideView(const vector<Point2f>& quad,
                                                         cv::Size imageSize,
                                                         QuadOrientation orientation);

   private:
    /**
     * Check if two lines are parallel within epsilon tolerance
     * Uses dot product of normalized direction vectors
     */
    static bool areLinesParallel(const Point2f& p1, const Point2f& p2, const Point2f& p3,
                                 const Point2f& p4, double epsilon);

    /**
     * Calculate angle in degrees between two edges
     * @param p1 Start point of first edge
     * @param p2 End point of first edge (shared corner)
     * @param p3 Start point of second edge (shared corner)
     * @param p4 End point of second edge
     * @return Angle in degrees between the two edges
     */
    static double angleBetweenEdges(const Point2f& p1, const Point2f& p2, const Point2f& p3,
                                    const Point2f& p4);

    /**
     * Calculate the apparent aspect ratio of the quad by dividing the average of the
     * horizontal edge lengths (top, bottom) by the average of the vertical edge lengths
     * (left, right).
     * @param quad Quadrilateral points (assumed ordered: top-left, top-right, bottom-right,
     * bottom-left)
     * @return The apparent aspect ratio (horizontal/vertical).
     */
    static double getApparentAspectRatio(const vector<Point2f>& quad);
};

#endif  // QUAD_ANALYSIS_HPP
