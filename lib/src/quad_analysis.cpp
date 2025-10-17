#include "quad_analysis.hpp"

#include <cmath>

#include "utilities.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

bool QuadAnalysis::areLinesParallel(const Point2f& p1, const Point2f& p2, const Point2f& p3,
                                    const Point2f& p4, double epsilon) {
    // Direction vectors
    Point2f dir1 = p2 - p1;
    Point2f dir2 = p4 - p3;

    // Normalize direction vectors
    float len1 = std::sqrt(dir1.x * dir1.x + dir1.y * dir1.y);
    float len2 = std::sqrt(dir2.x * dir2.x + dir2.y * dir2.y);

    if (len1 < 1e-6 || len2 < 1e-6) {
        LOGI("[QuadAnalysis] Degenerate line detected: len1=%.3f, len2=%.3f", len1, len2);
        return false;  // Degenerate line
    }

    dir1.x /= len1;
    dir1.y /= len1;
    dir2.x /= len2;
    dir2.y /= len2;

    // Dot product of normalized vectors
    // If parallel, dot product should be ±1
    float dotProduct = dir1.x * dir2.x + dir1.y * dir2.y;
    float absDot = std::abs(dotProduct);
    float deviation = 1.0 - absDot;

    bool isParallel = deviation < epsilon;
    LOGI(
        "[QuadAnalysis] Parallel check: dotProduct=%.4f, absDot=%.4f, deviation=%.4f, "
        "epsilon=%.4f, result=%s",
        dotProduct, absDot, deviation, epsilon, isParallel ? "PARALLEL" : "NOT PARALLEL");

    // Check if close to 1 (parallel, same direction) or -1 (parallel, opposite direction)
    return isParallel;
}

double QuadAnalysis::angleBetweenEdges(const Point2f& p1, const Point2f& p2, const Point2f& p3,
                                       const Point2f& p4) {
    // Direction vectors from the shared corner
    // Edge 1: p2 - p1 (vector from p1 to p2)
    // Edge 2: p4 - p3 (vector from p3 to p4)
    Point2f dir1 = p2 - p1;
    Point2f dir2 = p4 - p3;

    // Calculate lengths
    float len1 = std::sqrt(dir1.x * dir1.x + dir1.y * dir1.y);
    float len2 = std::sqrt(dir2.x * dir2.x + dir2.y * dir2.y);

    if (len1 < 1e-6 || len2 < 1e-6) {
        LOGI("[QuadAnalysis] angleBetweenEdges: Degenerate edge detected: len1=%.3f, len2=%.3f",
             len1, len2);
        return 0.0;  // Degenerate edge
    }

    // Normalize direction vectors
    dir1.x /= len1;
    dir1.y /= len1;
    dir2.x /= len2;
    dir2.y /= len2;

    // Dot product of normalized vectors: cos(θ) = v1 · v2
    float dotProduct = dir1.x * dir2.x + dir1.y * dir2.y;

    // Clamp to [-1, 1] to handle numerical errors
    dotProduct = std::max(-1.0f, std::min(1.0f, dotProduct));

    // Calculate angle in radians, then convert to degrees
    float angleRadians = std::acos(dotProduct);
    float angleDegrees = angleRadians * 180.0f / M_PI;

    LOGI("[QuadAnalysis] angleBetweenEdges: dotProduct=%.4f, angle=%.2f°", dotProduct,
         angleDegrees);

    return angleDegrees;
}

bool QuadAnalysis::topBottomParallel(const vector<Point2f>& quad, double epsilon) {
    if (quad.size() != 4) return false;

    LOGI("[QuadAnalysis] Checking top/bottom parallel:");
    LOGI("  Top line: (%.1f,%.1f) to (%.1f,%.1f)", quad[0].x, quad[0].y, quad[1].x, quad[1].y);
    LOGI("  Bottom line: (%.1f,%.1f) to (%.1f,%.1f)", quad[3].x, quad[3].y, quad[2].x, quad[2].y);

    // Top line: quad[0] to quad[1]
    // Bottom line: quad[3] to quad[2]
    bool result = areLinesParallel(quad[0], quad[1], quad[3], quad[2], epsilon);
    LOGI("  Top/Bottom parallel result: %s", result ? "TRUE" : "FALSE");
    return result;
}

bool QuadAnalysis::leftRightParallel(const vector<Point2f>& quad, double epsilon) {
    if (quad.size() != 4) return false;

    LOGI("[QuadAnalysis] Checking left/right parallel:");
    LOGI("  Left line: (%.1f,%.1f) to (%.1f,%.1f)", quad[0].x, quad[0].y, quad[3].x, quad[3].y);
    LOGI("  Right line: (%.1f,%.1f) to (%.1f,%.1f)", quad[1].x, quad[1].y, quad[2].x, quad[2].y);

    // Left line: quad[0] to quad[3]
    // Right line: quad[1] to quad[2]
    bool result = areLinesParallel(quad[0], quad[3], quad[1], quad[2], epsilon);
    LOGI("  Left/Right parallel result: %s", result ? "TRUE" : "FALSE");
    return result;
}

double QuadAnalysis::topRightRatio(const vector<Point2f>& quad) {
    if (quad.size() != 4) return 0.0;

    // Top line length: distance from quad[0] to quad[1]
    float topLength = cv::norm(quad[1] - quad[0]);

    // Right line length: distance from quad[1] to quad[2]
    float rightLength = cv::norm(quad[2] - quad[1]);

    if (rightLength < 1e-6) {
        LOGI("[QuadAnalysis] topRightRatio: rightLength too small (%.6f), returning 0.0",
             rightLength);
        return 0.0;
    }

    double ratio = topLength / rightLength;
    LOGI("[QuadAnalysis] topRightRatio: topLength=%.1f, rightLength=%.1f, ratio=%.3f", topLength,
         rightLength, ratio);
    return ratio;
}

double QuadAnalysis::topLeftRatio(const vector<Point2f>& quad) {
    if (quad.size() != 4) return 0.0;

    // Top line length: distance from quad[0] to quad[1]
    float topLength = cv::norm(quad[1] - quad[0]);

    // Left line length: distance from quad[0] to quad[3]
    float leftLength = cv::norm(quad[3] - quad[0]);

    if (leftLength < 1e-6) {
        LOGI("[QuadAnalysis] topLeftRatio: leftLength too small (%.6f), returning 0.0", leftLength);
        return 0.0;
    }

    double ratio = topLength / leftLength;
    LOGI("[QuadAnalysis] topLeftRatio: topLength=%.1f, leftLength=%.1f, ratio=%.3f", topLength,
         leftLength, ratio);
    return ratio;
}

QuadOrientation QuadAnalysis::orientation(const vector<Point2f>& quad) {
    if (quad.size() != 4) {
        LOGI("[QuadAnalysis] Invalid quad size: %zu (expected 4), returning OTHER", quad.size());
        return OTHER;
    }

    LOGI("[QuadAnalysis] ========== QUAD ORIENTATION ANALYSIS ==========");

    // Verify ordering by checking which point is actually where
    float centroidX = (quad[0].x + quad[1].x + quad[2].x + quad[3].x) / 4.0f;
    float centroidY = (quad[0].y + quad[1].y + quad[2].y + quad[3].y) / 4.0f;
    LOGI("[QuadAnalysis] Quad centroid: (%.1f, %.1f)", centroidX, centroidY);

    LOGI("[QuadAnalysis] Quad points (ordered counter-clockwise from atan2 sort):");
    for (int i = 0; i < 4; i++) {
        float angle = atan2(quad[i].y - centroidY, quad[i].x - centroidX) * 180.0f / M_PI;
        bool isLeft = quad[i].x < centroidX;
        bool isTop = quad[i].y < centroidY;
        const char* position =
            isTop ? (isLeft ? "Top-left" : "Top-right") : (isLeft ? "Bottom-left" : "Bottom-right");
        LOGI("  [%d] %s: (%.1f, %.1f) angle=%.1f°", i, position, quad[i].x, quad[i].y, angle);
    }

    // Calculate all 4 edge lengths for debugging
    LOGI("[QuadAnalysis] === GEOMETRIC ANALYSIS ===");
    float topLen = cv::norm(quad[1] - quad[0]);
    float rightLen = cv::norm(quad[2] - quad[1]);
    float bottomLen = cv::norm(quad[3] - quad[2]);
    float leftLen = cv::norm(quad[0] - quad[3]);

    LOGI("[QuadAnalysis] Edge lengths:");
    LOGI("  Top (0->1):    %.1f px", topLen);
    LOGI("  Right (1->2):  %.1f px", rightLen);
    LOGI("  Bottom (2->3): %.1f px", bottomLen);
    LOGI("  Left (3->0):   %.1f px", leftLen);

    // Calculate bounding box
    float minX = std::min({quad[0].x, quad[1].x, quad[2].x, quad[3].x});
    float maxX = std::max({quad[0].x, quad[1].x, quad[2].x, quad[3].x});
    float minY = std::min({quad[0].y, quad[1].y, quad[2].y, quad[3].y});
    float maxY = std::max({quad[0].y, quad[1].y, quad[2].y, quad[3].y});
    float bboxWidth = maxX - minX;
    float bboxHeight = maxY - minY;

    LOGI("[QuadAnalysis] Bounding box: %.1fx%.1f (WxH), aspect=%.3f (W/H)", bboxWidth, bboxHeight,
         bboxWidth / bboxHeight);

    // Average opposite edges to understand quad shape
    float avgHoriz = (topLen + bottomLen) / 2.0f;
    float avgVert = (leftLen + rightLen) / 2.0f;
    LOGI("[QuadAnalysis] Average edges: horiz=%.1f, vert=%.1f, ratio=%.3f (horiz/vert)", avgHoriz,
         avgVert, avgHoriz / avgVert);

    bool tbParallel = topBottomParallel(quad);
    bool lrParallel = leftRightParallel(quad);
    double apparentAspectRatio = getApparentAspectRatio(quad);

    LOGI("[QuadAnalysis] Analysis results:");
    LOGI("  Top/Bottom parallel: %s", tbParallel ? "TRUE" : "FALSE");
    LOGI("  Left/Right parallel: %s", lrParallel ? "TRUE" : "FALSE");
    LOGI("  Apparent Aspect Ratio (H/V): %.3f (threshold: 1.75)", apparentAspectRatio);

    QuadOrientation result;

    // Determine orientation based on criteria
    if (tbParallel && lrParallel) {
        // Check right angles at corners for TOP_DOWN
        // Top-left corner (quad[0]): angle between top edge (0→1) and left edge (0→3)
        double topLeftAngle = angleBetweenEdges(quad[0], quad[1], quad[0], quad[3]);
        double topRightAngle = angleBetweenEdges(quad[1], quad[0], quad[1], quad[2]);
        double bottomRightAngle = angleBetweenEdges(quad[2], quad[1], quad[2], quad[3]);
        double bottomLeftAngle = angleBetweenEdges(quad[3], quad[0], quad[3], quad[2]);

        const double RIGHT_ANGLE = 90.0;
        const double ANGLE_TOLERANCE = 3.0;

        double topLeftDiff = std::abs(topLeftAngle - RIGHT_ANGLE);
        double topRightDiff = std::abs(topRightAngle - RIGHT_ANGLE);
        double bottomRightDiff = std::abs(bottomRightAngle - RIGHT_ANGLE);
        double bottomLeftDiff = std::abs(bottomLeftAngle - RIGHT_ANGLE);

        LOGI("[QuadAnalysis] TOP_DOWN angle check:");
        LOGI("  Top-left corner angle: %.2f° (diff from 90°: %.2f°)", topLeftAngle, topLeftDiff);
        LOGI("  Top-right corner angle: %.2f° (diff from 90°: %.2f°)", topRightAngle, topRightDiff);
        LOGI("  Bottom-right corner angle: %.2f° (diff from 90°: %.2f°)", bottomRightAngle,
             bottomRightDiff);
        LOGI("  Bottom-left corner angle: %.2f° (diff from 90°: %.2f°)", bottomLeftAngle,
             bottomLeftDiff);
        LOGI("  Tolerance: %.2f°", ANGLE_TOLERANCE);

        if (topLeftDiff < ANGLE_TOLERANCE && topRightDiff < ANGLE_TOLERANCE &&
            bottomRightDiff < ANGLE_TOLERANCE && bottomLeftDiff < ANGLE_TOLERANCE) {
            result = TOP_DOWN;
            LOGI(
                "[QuadAnalysis] Decision: TOP_DOWN (both TB and LR parallel, right angles at "
                "corners)");
        } else {
            result = OTHER;
            LOGI(
                "[QuadAnalysis] Decision: OTHER (TB and LR parallel, but angles not right: TL "
                "diff=%.2f°, TR diff=%.2f°, BR diff=%.2f°, BL diff=%.2f°)",
                topLeftDiff, topRightDiff, bottomRightDiff, bottomLeftDiff);
        }
    } else if (tbParallel && !lrParallel) {
        // Top/Bottom edges are parallel - viewing from side with horizontal perspective

        // Check if left and right edges are approximately equal length
        // This should be true for valid side views since they represent the same dimension of the table
        float lrLengthDiff = std::abs(leftLen - rightLen);
        float avgLR = (leftLen + rightLen) / 2.0f;
        float lrLengthRatio = avgLR > 0 ? lrLengthDiff / avgLR : 0.0f;
        const float LR_LENGTH_TOLERANCE = 0.15f;  // 15% tolerance
        bool lrLengthsEqual = lrLengthRatio < LR_LENGTH_TOLERANCE;

        LOGI("[QuadAnalysis] Left/Right edge length check:");
        LOGI("  Left length: %.1f px", leftLen);
        LOGI("  Right length: %.1f px", rightLen);
        LOGI("  Difference: %.1f px", lrLengthDiff);
        LOGI("  Relative difference: %.1f%% (tolerance: %.1f%%)", lrLengthRatio * 100.0f, LR_LENGTH_TOLERANCE * 100.0f);
        LOGI("  Lengths approximately equal: %s", lrLengthsEqual ? "TRUE" : "FALSE");

        // Calculate all four corner angles
        double topLeftAngle = angleBetweenEdges(quad[0], quad[1], quad[0], quad[3]);
        double topRightAngle = angleBetweenEdges(quad[1], quad[0], quad[1], quad[2]);
        double bottomRightAngle = angleBetweenEdges(quad[2], quad[1], quad[2], quad[3]);
        double bottomLeftAngle = angleBetweenEdges(quad[3], quad[0], quad[3], quad[2]);

        LOGI("[QuadAnalysis] Side view angle check:");
        LOGI("  Top-left: %.2f°, Top-right: %.2f° (must be > 90°)", topLeftAngle, topRightAngle);
        LOGI("  Bottom-left: %.2f°, Bottom-right: %.2f° (must be < 90°)", bottomLeftAngle,
             bottomRightAngle);

        bool topAnglesObtuse = topLeftAngle > 90.0 && topRightAngle > 90.0;
        bool bottomAnglesAcute = bottomLeftAngle < 90.0 && bottomRightAngle < 90.0;

        // Both LONG_SIDE and SHORT_SIDE require approximately equal left/right lengths
        if (!lrLengthsEqual) {
            result = OTHER;
            LOGI(
                "[QuadAnalysis] Decision: OTHER (TB parallel, but left/right lengths not equal: "
                "diff=%.1f%%, tolerance=%.1f%%)",
                lrLengthRatio * 100.0f, LR_LENGTH_TOLERANCE * 100.0f);
        } else if (apparentAspectRatio >= 1.75) {
            result = LONG_SIDE;
            LOGI(
                "[QuadAnalysis] Decision: LONG_SIDE (TB parallel, LR not parallel, LR lengths equal, "
                "ratio %.3f >= 1.75)",
                apparentAspectRatio);
        } else if (topAnglesObtuse && bottomAnglesAcute) {
            result = SHORT_SIDE;
            LOGI(
                "[QuadAnalysis] Decision: SHORT_SIDE (TB parallel, LR lengths equal, ratio %.3f < 1.75, "
                "angles correct)",
                apparentAspectRatio);
        } else {
            result = OTHER;
            LOGI(
                "[QuadAnalysis] Decision: OTHER (TB parallel, LR lengths equal, but failed angle checks: "
                "top obtuse=%s, bottom acute=%s)",
                topAnglesObtuse ? "PASS" : "FAIL", bottomAnglesAcute ? "PASS" : "FAIL");
        }
    } else if (!tbParallel && lrParallel) {
        // Left/Right edges are parallel, but top/bottom are not.
        // Per the requirement, this cannot be a SHORT_SIDE or LONG_SIDE view.
        result = OTHER;
        LOGI(
            "[QuadAnalysis] Decision: OTHER (LR parallel, but TB not parallel, which is not a "
            "valid SHORT_SIDE or LONG_SIDE view)");
    } else {
        result = OTHER;
        LOGI("[QuadAnalysis] Decision: OTHER (neither pair of opposite edges is parallel)");
    }

    LOGI("[QuadAnalysis] Final orientation: %s", orientationToString(result).c_str());
    LOGI("[QuadAnalysis] ================================================");

    return result;
}

double QuadAnalysis::getApparentAspectRatio(const vector<Point2f>& quad) {
    if (quad.size() != 4) return 0.0;

    float topLen = cv::norm(quad[1] - quad[0]);
    float bottomLen = cv::norm(quad[2] - quad[3]);
    float leftLen = cv::norm(quad[3] - quad[0]);
    float rightLen = cv::norm(quad[2] - quad[1]);

    float avgHorizontal = (topLen + bottomLen) / 2.0f;
    float avgVertical = (leftLen + rightLen) / 2.0f;

    if (avgVertical < 1e-6) {
        LOGI("[QuadAnalysis] getApparentAspectRatio: avgVertical too small (%.6f), returning 0.0",
             avgVertical);
        return 0.0;
    }

    double ratio = avgHorizontal / avgVertical;
    LOGI("[QuadAnalysis] getApparentAspectRatio: avgHorizontal=%.1f, avgVertical=%.1f, ratio=%.3f",
         avgHorizontal, avgVertical, ratio);
    return ratio;
}

string QuadAnalysis::orientationToString(QuadOrientation orientation) {
    switch (orientation) {
        case SHORT_SIDE:
            return "SHORT_SIDE";
        case LONG_SIDE:
            return "LONG_SIDE";
        case TOP_DOWN:
            return "TOP_DOWN";
        case OTHER:
            return "OTHER";
        default:
            return "UNKNOWN";
    }
}

ViewValidation QuadAnalysis::validateLandscapeShortSideView(const vector<Point2f>& quad,
                                                            cv::Size imageSize,
                                                            QuadOrientation orientation) {
    ViewValidation result;
    result.isValid = true;
    result.errorMessage = "";

    LOGI("[QuadAnalysis] ========== VIEW VALIDATION ==========");
    LOGI("[QuadAnalysis] Image size: %dx%d (WxH)", imageSize.width, imageSize.height);
    LOGI("[QuadAnalysis] Orientation: %s", orientationToString(orientation).c_str());

    // Check 1: Landscape orientation (width > height)
    result.isLandscape = imageSize.width > imageSize.height;
    LOGI("[QuadAnalysis] Landscape check: %s (width=%d, height=%d)",
         result.isLandscape ? "PASS" : "FAIL", imageSize.width, imageSize.height);

    if (!result.isLandscape) {
        result.isValid = false;
        if (!result.errorMessage.empty()) result.errorMessage += "; ";
        result.errorMessage += "Image is not landscape (portrait detected)";
    }

    // Check 2: 16:9 aspect ratio (~1.778)
    result.imageAspectRatio = (float)imageSize.width / (float)imageSize.height;
    const float TARGET_ASPECT = 16.0f / 9.0f;  // 1.778
    const float ASPECT_TOLERANCE = 0.1f;
    float aspectDiff = std::abs(result.imageAspectRatio - TARGET_ASPECT);
    result.isCorrectAspectRatio = aspectDiff < ASPECT_TOLERANCE;

    LOGI(
        "[QuadAnalysis] Aspect ratio check: %s (actual=%.3f, target=%.3f, diff=%.3f, "
        "tolerance=%.3f)",
        result.isCorrectAspectRatio ? "PASS" : "FAIL", result.imageAspectRatio, TARGET_ASPECT,
        aspectDiff, ASPECT_TOLERANCE);

    if (!result.isCorrectAspectRatio) {
        result.isValid = false;
        if (!result.errorMessage.empty()) result.errorMessage += "; ";
        char buffer[256];
        snprintf(buffer, sizeof(buffer), "Image aspect ratio %.3f is not 16:9 (expected ~%.2f)",
                 result.imageAspectRatio, TARGET_ASPECT);
        result.errorMessage += buffer;
    }

    // Check 3: Camera at SHORT side of table (SHORT_SIDE orientation)
    result.isShortSideView = (orientation == SHORT_SIDE);
    LOGI("[QuadAnalysis] Short-side view check: %s (orientation=%s, expected=SHORT_SIDE)",
         result.isShortSideView ? "PASS" : "FAIL", orientationToString(orientation).c_str());

    if (!result.isShortSideView) {
        result.isValid = false;
        if (!result.errorMessage.empty()) result.errorMessage += "; ";
        char buffer[256];
        snprintf(buffer, sizeof(buffer),
                 "Table not viewed from short side (orientation: %s, expected: SHORT_SIDE)",
                 orientationToString(orientation).c_str());
        result.errorMessage += buffer;
    }

    // Final result
    LOGI("[QuadAnalysis] === VALIDATION RESULT: %s ===", result.isValid ? "VALID" : "INVALID");
    if (!result.isValid) {
        LOGI("[QuadAnalysis] Error: %s", result.errorMessage.c_str());
    }
    LOGI("[QuadAnalysis] ============================================");

    return result;
}
