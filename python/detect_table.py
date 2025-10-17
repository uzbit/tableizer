# detect_table.py
# ---------------------------------------------------------------
"""Pool table detection, ball detection, and coordinate transformation."""

import csv
import glob
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from utilities import (
    load_detection_model,
    calculate_ball_pixel_size,
    order_quad,
)
from tableizer_ffi import detect_table_cpp, transform_points_cpp


# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

MODEL_NAMES = ["baseline", "combined", "combined2", "combined3"]
SHOTSTUDIO_BG_PATH = "../data/shotstudio_table_felt_only.png"

# Processing parameters
SHOTSTUDIO_SIZE = (840, 1680)

# Detection parameters
CONF_THRESHOLD = 0.6

# Visualization parameters
DO_PLOT = True  # Set to False to disable all plots

# Visualization constants
CIRCLE_RADIUS = 15
LINE_THICKNESS = 3
MIN_BALL_SIZE = 8
ROTATION_THRESHOLD_RATIO = 1.75

# Ball colors (BGR format) - ordered by class: ["black", "cue", "solid", "stripe"]
BALL_COLORS = [
    (0, 0, 0),          # Class 0: Black
    (255, 255, 255),    # Class 1: Cue (White)
    (0, 0, 255),        # Class 2: Solid (Red)
    (0, 255, 255),      # Class 3: Stripe (Yellow)
]

# Global ShotStudio background (loaded once)
_SHOTSTUDIO_BG = None


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _check_rotation_needed(ordered_quad):
    """Check if table needs rotation from landscape to portrait.

    Args:
        ordered_quad (np.ndarray): Ordered quadrilateral points (4, 2)

    Returns:
        bool: True if rotation from landscape to portrait is needed
    """
    top_length = np.linalg.norm(ordered_quad[1] - ordered_quad[0])
    right_length = np.linalg.norm(ordered_quad[2] - ordered_quad[1])
    return top_length > right_length * ROTATION_THRESHOLD_RATIO


# ============================================================================
# TABLE DETECTION FUNCTIONS
# ============================================================================

def detect_table_and_validate(img):
    """Detect table quadrilateral and validate results.

    Args:
        img (np.ndarray): Input image

    Returns:
        np.ndarray or None: quad_points - numpy.ndarray of shape (4, 2) with quad coordinates
        Returns None if detection failed
    """
    print("Using C++ table detection...")
    detection_result = detect_table_cpp(img, rotation_degrees=0)

    if not detection_result or "quad_points" not in detection_result:
        print("C++ table detection failed!")
        return None

    quad_points_list = detection_result["quad_points"]
    if len(quad_points_list) != 4:
        print(f"Invalid quad points count: {len(quad_points_list)}")
        return None

    # Convert from list format to numpy array - these are in FULL resolution coordinates
    quad = np.array(quad_points_list, dtype=np.float32)

    print(f"C++ table detection found quad (full res): {quad}")

    return quad


def extract_table_region(original_img, quad):
    """Extract table region using quad and transform to portrait mode (840x1680).

    Args:
        original_img (np.ndarray): Original input image
        quad (np.ndarray): Quadrilateral points of table (4, 2)

    Returns:
        np.ndarray: Extracted and warped table image in portrait mode
    """
    ordered_quad = order_quad(quad)
    needs_rotation = _check_rotation_needed(ordered_quad)

    out_w, out_h = SHOTSTUDIO_SIZE

    if needs_rotation:
        # Table is landscape in image, needs rotation to portrait
        # First warp to landscape (1680x840), then rotate to portrait
        landscape_dst = np.array(
            [[0, 0], [out_h - 1, 0], [out_h - 1, out_w - 1], [0, out_w - 1]],
            dtype=np.float32,
        )

        h_landscape = cv2.getPerspectiveTransform(ordered_quad, landscape_dst)

        # Add 90° CCW rotation: landscape (1680x840) -> portrait (840x1680)
        rot = np.array([[0, 1, 0], [-1, 0, out_h - 1], [0, 0, 1]], np.float32)
        h_final = rot @ h_landscape
        extracted = cv2.warpPerspective(original_img, h_final, (out_w, out_h))
    else:
        # Table is already portrait in image, direct warp
        portrait_dst = np.array(
            [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
            dtype=np.float32,
        )

        h = cv2.getPerspectiveTransform(ordered_quad, portrait_dst)
        extracted = cv2.warpPerspective(original_img, h, (out_w, out_h))

    top_length = np.linalg.norm(ordered_quad[1] - ordered_quad[0])
    right_length = np.linalg.norm(ordered_quad[2] - ordered_quad[1])
    print(
        f"Table extraction: needs_rotation={needs_rotation}, "
        f"top_length={top_length:.1f}, right_length={right_length:.1f}"
    )
    return extracted


# ============================================================================
# BALL DETECTION FUNCTIONS
# ============================================================================

def get_balls_from_image(image_input, model_path, quad=None):
    """Extract ball centers, classes, and confidences from an image using YOLO detection.

    Args:
        image_input: Either a file path (str) or image array (np.ndarray)
        model_path (str): Path to the YOLO model weights
        quad (np.ndarray or None): Quadrilateral points (4, 2) to mask the image. If provided,
                                    only the region inside the quad will be used for detection.

    Returns:
        tuple: (ball_centers, ball_classes, ball_confidences) where:
            - ball_centers: np.ndarray of shape (N, 2) with ball center coordinates
            - ball_classes: np.ndarray of shape (N,) with ball class IDs
            - ball_confidences: np.ndarray of shape (N,) with detection confidences
        Returns (None, None, None) if no balls detected
    """
    # Apply mask if quad is provided (same as C++ createMaskedImage)
    if quad is not None and isinstance(image_input, np.ndarray):
        # Create a zeros mask
        mask = np.zeros(image_input.shape[:2], dtype=np.uint8)

        # Fill the quad region with white
        quad_int = np.round(quad).astype(np.int32)
        cv2.fillConvexPoly(mask, quad_int, 255)

        # Apply mask to image (zero out everything outside quad)
        image_for_detection = cv2.bitwise_and(image_input, image_input, mask=mask)

        print(f"Applied mask using quad points for ball detection")
    else:
        image_for_detection = image_input

    model = load_detection_model(model_path)

    results = model(image_for_detection)
    boxes = results[0].boxes

    if len(boxes) == 0:
        return None, None, None

    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    cls_ids = boxes.cls.cpu().numpy()

    x1, y1, x2, y2 = xyxy.T
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = x2 - x1, y2 - y1

    print(f"Raw detections before filtering: {len(confs)}")
    print(f"Confidence values: {confs}")
    print(f"Confidence threshold: {CONF_THRESHOLD}")

    valid_mask = confs >= CONF_THRESHOLD
    print(f"Detections passing confidence threshold: {valid_mask.sum()}")

    if not valid_mask.any():
        print("No detections passed confidence threshold!")
        return None, None, None

    dets = np.column_stack(
        [
            cx[valid_mask],
            cy[valid_mask],
            w[valid_mask],
            h[valid_mask],
            confs[valid_mask],
            cls_ids[valid_mask],
        ]
    )

    print(f"Total detections after post-processing: {len(dets)}")
    print(f"Detection classes: {dets[:, 5] if len(dets) > 0 else 'none'}")

    # Filter out class 4 (non-ball class)
    balls = dets[dets[:, 5] != 4]
    print(f"Balls after filtering class 4: {len(balls)}")
    print(f"Ball classes: {balls[:, 5] if len(balls) > 0 else 'none'}")

    if balls.size == 0:
        print("No balls detected.")
        return None, None, None

    ball_centers = np.array([[b[0], b[1]] for b in balls])
    ball_classes = balls[:, -1]
    ball_confidences = balls[:, 4]

    print(f"Ball confidences: {ball_confidences}")
    print(f"Final ball count: {len(ball_centers)}")

    return ball_centers, ball_classes, ball_confidences


# ============================================================================
# TRANSFORMATION FUNCTIONS
# ============================================================================

def transform_balls_for_extracted_table(ball_centers, quad):
    """Transform ball positions to match the extracted table region coordinates.

    Args:
        ball_centers (np.ndarray): Ball center coordinates in original image (N, 2)
        quad (np.ndarray): Quadrilateral points of table (4, 2)

    Returns:
        np.ndarray: Transformed ball positions in extracted table coordinates (N, 2)
    """
    ordered_quad = order_quad(quad)
    needs_rotation = _check_rotation_needed(ordered_quad)

    out_w, out_h = SHOTSTUDIO_SIZE

    ball_points = np.array(ball_centers, dtype=np.float32)

    if needs_rotation:
        # Table is landscape in image, needs rotation to portrait
        # First warp to landscape (1680x840), then rotate to portrait
        landscape_dst = np.array(
            [[0, 0], [out_h - 1, 0], [out_h - 1, out_w - 1], [0, out_w - 1]],
            dtype=np.float32,
        )

        h_landscape = cv2.getPerspectiveTransform(ordered_quad, landscape_dst)

        # Add 90° CCW rotation: landscape (1680x840) -> portrait (840x1680)
        rot = np.array([[0, 1, 0], [-1, 0, out_h - 1], [0, 0, 1]], np.float32)
        h_final = rot @ h_landscape
    else:
        # Table is already portrait in image, direct warp
        portrait_dst = np.array(
            [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
            dtype=np.float32,
        )

        h_final = cv2.getPerspectiveTransform(ordered_quad, portrait_dst)

    # Transform the ball points
    ball_points_homogeneous = np.ones((len(ball_points), 3), dtype=np.float32)
    ball_points_homogeneous[:, :2] = ball_points

    # Apply transformation
    transformed_homogeneous = (h_final @ ball_points_homogeneous.T).T

    # Convert back to 2D coordinates
    transformed_points = (
        transformed_homogeneous[:, :2] / transformed_homogeneous[:, 2:3]
    )

    return transformed_points


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def visualize_quad_detection(img, quad):
    """Show table quad detection on original image.

    Args:
        img (np.ndarray): Input image
        quad (np.ndarray): Quadrilateral points (4, 2)
    """
    vis = img.copy()
    print(f"Drawing quad: {quad}")
    cv2.polylines(vis, [quad.astype(int)], True, (0, 0, 255), 2)
    for p in quad.astype(int):
        cv2.circle(vis, tuple(p), 6, (0, 0, 255), -1)
    print("BL, TL, BR, TR  (pixel coords in full frame):\n", quad)
    if DO_PLOT:
        cv2.imshow("Quad via cell extremums", vis)


def show_detection_failed(img):
    """Show red X when detection fails.

    Args:
        img (np.ndarray): Input image
    """
    vis = img.copy()
    h, w = vis.shape[:2]
    # Draw from top-left to bottom-right
    cv2.line(vis, (0, 0), (w - 1, h - 1), (0, 0, 255), 2)
    # Draw from bottom-left to top-right
    cv2.line(vis, (0, h - 1), (w - 1, 0), (0, 0, 255), 2)
    if DO_PLOT:
        cv2.imshow("Red X", vis)
    print("NO QUAD FOUND!")


def draw_ball_overlay_on_image(
    ball_positions,
    ball_classes,
    background_img,
    title="Image with Ball Detections",
    use_circle_indicators=False,
):
    """Draw ball overlays on any image (ShotStudio, extracted table, or original image).

    Args:
        ball_positions (np.ndarray): Ball positions (N, 2)
        ball_classes (np.ndarray): Ball class IDs (N,)
        background_img (np.ndarray): Background image to draw on
        title (str): Window title for display
        use_circle_indicators (bool): If True, use circle outlines; if False, use filled circles

    Returns:
        np.ndarray: Image with ball overlays drawn
    """
    overlay = background_img.copy()

    if use_circle_indicators:
        # Use circle indicators like "Original image with Detections"
        for position, cls in zip(ball_positions, ball_classes):
            center = (int(round(position[0])), int(round(position[1])))
            color = BALL_COLORS[int(cls) % len(BALL_COLORS)]

            # Draw circle outline
            cv2.circle(overlay, center, CIRCLE_RADIUS, color, LINE_THICKNESS)
            # Add class label
            cv2.putText(
                overlay,
                f"{int(cls)}",
                (center[0] + 8, center[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
                cv2.LINE_AA,
            )
    else:
        # Use filled circles for ShotStudio-style visualization
        # Calculate ball size based on table dimensions
        ball_dia_px = calculate_ball_pixel_size(background_img, table_size=78)
        ball_dia_px = max(ball_dia_px, MIN_BALL_SIZE)

        for position, cls in zip(ball_positions, ball_classes):
            center = (int(round(position[0])), int(round(position[1])))
            color = BALL_COLORS[int(cls) % len(BALL_COLORS)]

            # Draw filled circle for ball
            cv2.circle(overlay, center, ball_dia_px // 2, color, -1)
            # Draw border
            cv2.circle(overlay, center, ball_dia_px // 2, (0, 0, 0), 2)

            # Add class label
            cv2.putText(
                overlay,
                f"{int(cls)}",
                (center[0] + 8, center[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )

    if DO_PLOT:
        cv2.imshow(title, overlay)
    return overlay


# ============================================================================
# SHOTSTUDIO BACKGROUND MANAGEMENT
# ============================================================================

def get_shotstudio_background():
    """Load and prepare ShotStudio background image (cached).

    Returns:
        np.ndarray or None: Prepared ShotStudio background image, or None if failed
    """
    global _SHOTSTUDIO_BG

    if _SHOTSTUDIO_BG is None:
        print("Loading ShotStudio background...")
        bg = cv2.imread(SHOTSTUDIO_BG_PATH)
        if bg is None:
            print(f"Could not load ShotStudio background: {SHOTSTUDIO_BG_PATH}")
            return None

        bg_h, bg_w = bg.shape[:2]
        print(f"Original ShotStudio background size: {bg_w}x{bg_h}")

        # Rotate to portrait if needed to match SHOTSTUDIO_SIZE (840x1680)
        target_w, target_h = SHOTSTUDIO_SIZE
        if bg_w > bg_h and target_h > target_w:
            print("Rotating ShotStudio background from landscape to portrait")
            bg = cv2.rotate(bg, cv2.ROTATE_90_COUNTERCLOCKWISE)
            bg_h, bg_w = bg.shape[:2]
            print(f"Rotated ShotStudio background size: {bg_w}x{bg_h}")

        # Resize to target size
        if (bg_w != target_w) or (bg_h != target_h):
            print(f"Resizing ShotStudio background to {target_w}x{target_h}")
            bg = cv2.resize(bg, (target_w, target_h))

        _SHOTSTUDIO_BG = bg
        print("ShotStudio background prepared and cached")

    return _SHOTSTUDIO_BG.copy()  # Return a copy to avoid modifying the cached version


# ============================================================================
# LOGGING FUNCTIONS
# ============================================================================

def print_model_banner(model_name):
    """Print a completion banner for a model.

    Args:
        model_name (str): Name of the model
    """
    print("\n" + "=" * 100)
    print("=" * 100)
    print(f"{'':^100}")
    print(f"MODEL: {model_name.upper()} - COMPLETE".center(100))
    print(f"{'':^100}")
    print("=" * 100)
    print("=" * 100 + "\n")


def get_detection_stats(detection_data):
    """Calculate detection statistics from detection data.

    Args:
        detection_data (dict or None): Dictionary with 'classes', 'confidences', 'positions' keys, or None if no balls detected

    Returns:
        dict: Dictionary with detection statistics
    """
    if detection_data is None:
        return {
            'number_balls_detected': 0,
            'cue_detected': 0,
            'black_detected': 0,
            'num_solids': 0,
            'num_stripes': 0,
            'num_object_balls': 0,
        }

    ball_classes = detection_data['classes']
    ball_confidences = detection_data['confidences']

    # Count total balls
    num_balls = len(ball_classes)

    # Initialize counts
    cue_detected = 0
    black_detected = 0
    num_solids = 0
    num_stripes = 0
    num_object_balls = 0

    # Class 0 is black ball, Class 1 is cue ball
    # Class 2 is solid, Class 3 is stripe
    black_detected = 1 if 0 in ball_classes else 0
    cue_detected = 1 if 1 in ball_classes else 0
    num_solids = int(np.sum(ball_classes == 2))
    num_stripes = int(np.sum(ball_classes == 3))
    num_object_balls = num_solids + num_stripes

    return {
        'number_balls_detected': num_balls,
        'cue_detected': cue_detected,
        'black_detected': black_detected,
        'num_solids': num_solids,
        'num_stripes': num_stripes,
        'num_object_balls': num_object_balls,
    }


def compute_baseline_comparison(baseline_data, model_data):
    """Compute comparison metrics between baseline and another model's detections.

    Args:
        baseline_data (dict or None): Baseline detection data with 'positions', 'classes', 'confidences'
        model_data (dict or None): Model detection data with 'positions', 'classes', 'confidences'

    Returns:
        dict: Dictionary with comparison statistics vs baseline
    """
    # Handle cases where one or both models have no detections
    if baseline_data is None and model_data is None:
        return {
            'vs_baseline_ball_count_diff': 0,
            'vs_baseline_solid_diff': 0,
            'vs_baseline_stripe_diff': 0,
            'vs_baseline_cue_agreement': 1,
            'vs_baseline_black_agreement': 1,
            'vs_baseline_total_count_agreement': 1,
            'vs_baseline_position_overlap': 0,
            'vs_baseline_unique_detections': 0,
        }

    # Extract baseline stats
    baseline_classes = baseline_data['classes'] if baseline_data else np.array([])
    baseline_positions = baseline_data['positions'] if baseline_data else np.array([])

    baseline_count = len(baseline_classes)
    baseline_cue = 1 if 1 in baseline_classes else 0
    baseline_black = 1 if 0 in baseline_classes else 0
    baseline_solids = int(np.sum(baseline_classes == 2))
    baseline_stripes = int(np.sum(baseline_classes == 3))

    # Extract model stats
    model_classes = model_data['classes'] if model_data else np.array([])
    model_positions = model_data['positions'] if model_data else np.array([])

    model_count = len(model_classes)
    model_cue = 1 if 1 in model_classes else 0
    model_black = 1 if 0 in model_classes else 0
    model_solids = int(np.sum(model_classes == 2))
    model_stripes = int(np.sum(model_classes == 3))

    # Compute count differences
    ball_count_diff = model_count - baseline_count
    solid_diff = model_solids - baseline_solids
    stripe_diff = model_stripes - baseline_stripes

    # Compute agreements (1 if both agree, 0 if they disagree)
    cue_agreement = 1 if model_cue == baseline_cue else 0
    black_agreement = 1 if model_black == baseline_black else 0
    total_count_agreement = 1 if model_count == baseline_count else 0

    # Compute positional overlap (count balls within 50px threshold)
    position_overlap = 0
    unique_detections = model_count

    if baseline_count > 0 and model_count > 0:
        # For each model detection, find closest baseline detection
        matched_baseline_indices = set()
        matched_model_count = 0

        for i, model_pos in enumerate(model_positions):
            min_dist = float('inf')
            closest_baseline_idx = -1

            for j, baseline_pos in enumerate(baseline_positions):
                dist = np.linalg.norm(model_pos - baseline_pos)
                if dist < min_dist:
                    min_dist = dist
                    closest_baseline_idx = j

            # If within threshold (50 pixels) and same class, count as overlap
            if min_dist < 50 and closest_baseline_idx >= 0:
                if model_classes[i] == baseline_classes[closest_baseline_idx]:
                    position_overlap += 1
                    matched_baseline_indices.add(closest_baseline_idx)
                    matched_model_count += 1

        unique_detections = model_count - matched_model_count

    return {
        'vs_baseline_ball_count_diff': ball_count_diff,
        'vs_baseline_solid_diff': solid_diff,
        'vs_baseline_stripe_diff': stripe_diff,
        'vs_baseline_cue_agreement': cue_agreement,
        'vs_baseline_black_agreement': black_agreement,
        'vs_baseline_total_count_agreement': total_count_agreement,
        'vs_baseline_position_overlap': position_overlap,
        'vs_baseline_unique_detections': unique_detections,
    }


def write_results_to_xlsx(results_dict, output_file="detection_results.xlsx"):
    """Write detection results to Excel file in wide format with interleaved columns.

    Args:
        results_dict (dict): Dictionary with structure {image_name: {model_name: {stat_name: value}}}
        output_file (str): Output Excel file path

    Returns:
        bool: True if writing succeeded, False otherwise
    """
    try:
        # Build list of rows
        rows = []
        for image_name, model_results in results_dict.items():
            row = {'image_name': image_name}
            for model_name, stats in model_results.items():
                for stat_name, value in stats.items():
                    row[f'{model_name}_{stat_name}'] = value
            rows.append(row)

        # Create DataFrame
        df = pd.DataFrame(rows)

        # Collect all unique stat names across all models
        all_stat_names = set()
        first_image = next(iter(results_dict.values()))
        for model_name in MODEL_NAMES:
            if model_name in first_image:
                all_stat_names.update(first_image[model_name].keys())

        # Only use base stats, exclude comparison stats
        base_stats = [s for s in all_stat_names if not s.startswith('vs_baseline_')]

        # Build column order: image_name, then for each base stat show all models
        ordered_cols = ['image_name']

        # Add base stats columns (all models)
        for stat_name in sorted(base_stats):
            for model_name in MODEL_NAMES:
                col_name = f'{model_name}_{stat_name}'
                if col_name in df.columns:
                    ordered_cols.append(col_name)

        # Reorder columns
        df = df[ordered_cols]

        # Calculate averages for all numeric columns
        avg_row = {'image_name': 'AVERAGE'}
        for col in ordered_cols:
            if col != 'image_name':
                avg_row[col] = df[col].mean()

        # Calculate differences from baseline for base stats
        diff_row = {'image_name': 'DIFF FROM BASELINE'}
        for stat_name in sorted(base_stats):
            baseline_col = f'baseline_{stat_name}'
            if baseline_col in df.columns:
                baseline_avg = df[baseline_col].mean()
                for model_name in MODEL_NAMES:
                    if model_name != 'baseline':
                        model_col = f'{model_name}_{stat_name}'
                        if model_col in df.columns:
                            model_avg = df[model_col].mean()
                            diff_row[model_col] = model_avg - baseline_avg
                        else:
                            diff_row[model_col] = 0
                # Set baseline diff to 0
                diff_row[baseline_col] = 0

        # Append summary rows
        df = pd.concat([df, pd.DataFrame([avg_row, diff_row])], ignore_index=True)

        df.to_excel(output_file, index=False)

        print(f"Results written to {output_file}")
        return True

    except Exception as e:
        print(f"Error writing results to Excel: {e}")
        return False


# ============================================================================
# MAIN PROCESSING FUNCTIONS
# ============================================================================

def process_balls_and_visualize(img, quad, orig_quad_points, orig_img_size, model_path, model_name):
    """Detect balls, transform coordinates, and create visualizations.

    Args:
        img (np.ndarray): Input image
        quad (np.ndarray): Table quadrilateral points (4, 2)
        orig_quad_points (list): Original quad points as list of tuples
        orig_img_size (tuple): Original image size (width, height)
        model_path (str): Path to the YOLO model weights
        model_name (str): Name of the model being used

    Returns:
        dict or None: Detection data with keys 'positions', 'classes', 'confidences', or None if no balls detected
    """
    # Get ball detections with masking (same as C++ library)
    ball_centers, ball_classes, ball_confidences = get_balls_from_image(img, model_path, quad=quad)

    if ball_centers is None or len(ball_centers) == 0:
        print("No balls detected")
        return None

    print(f"Found {len(ball_centers)} balls")

    print(f"Ball centers in full resolution coordinates: {ball_centers}")
    print(f"Quad points: {orig_quad_points}")
    print(f"Original image size: {orig_img_size}")

    # Transform ball positions directly to ShotStudio coordinates
    print(f"Target ShotStudio size: {SHOTSTUDIO_SIZE}")

    transformed_points = transform_points_cpp(
        ball_centers.tolist(), orig_quad_points, orig_img_size, SHOTSTUDIO_SIZE
    )

    if not transformed_points:
        print("C++ coordinate transformation failed")
        return None

    # Print transformed coordinates
    transformed_coords = [[pt["x"], pt["y"]] for pt in transformed_points]
    print(f"Transformed ball positions in ShotStudio coordinates: {transformed_coords}")

    # Draw overlay on original image (circle indicators with quad)
    temp_img = img.copy()
    cv2.polylines(temp_img, [quad.astype(int)], True, (0, 255, 0), LINE_THICKNESS)
    draw_ball_overlay_on_image(
        ball_centers,
        ball_classes,
        temp_img,
        f"[{model_name.upper()}] Original Image with Detections",
        use_circle_indicators=True,
    )

    # # Extract table region in portrait mode (same size as ShotStudio)
    # extracted_table = extract_table_region(img, quad)

    # # Transform ball positions for the extracted table region
    # extracted_ball_positions = transform_balls_for_extracted_table(ball_centers, quad)

    # # Draw balls on extracted table region (circle indicators)
    # draw_ball_overlay_on_image(
    #     extracted_ball_positions,
    #     ball_classes,
    #     extracted_table,
    #     f"[{model_name.upper()}] Extracted Table with Ball Detections",
    #     use_circle_indicators=True,
    # )

    # # Get prepared ShotStudio background (loaded and cached)
    # shotstudio_bg = get_shotstudio_background()
    # if shotstudio_bg is None:
    #     return {
    #         'positions': ball_centers,
    #         'classes': ball_classes,
    #         'confidences': ball_confidences
    #     }

    # # Draw balls on ShotStudio background with same positions and style as extracted table
    # draw_ball_overlay_on_image(
    #     extracted_ball_positions,
    #     ball_classes,
    #     shotstudio_bg,
    #     f"[{model_name.upper()}] ShotStudio with Ball Detections",
    #     use_circle_indicators=True,
    # )

    # Wait for user input to close windows
    if DO_PLOT:
        cv2.waitKey(0)

    # Return full detection data
    return {
        'positions': ball_centers,
        'classes': ball_classes,
        'confidences': ball_confidences
    }


def run_detect(img, quad, model_name):
    """Run ball detection on an image with a specific model.

    Args:
        img (np.ndarray): Input image
        quad (np.ndarray): Table quadrilateral points (4, 2)
        model_name (str): Name of the model to use

    Returns:
        dict or None: Detection data with keys 'positions', 'classes', 'confidences', or None if detection failed
    """
    model_path = f"/Users/uzbit/Documents/projects/tableizer/tableizer/{model_name}/weights/best.pt"

    # Check if ShotStudio background exists before processing
    if not os.path.exists(SHOTSTUDIO_BG_PATH):
        print(f"ShotStudio background not found at: {SHOTSTUDIO_BG_PATH}")
        return None

    # Process balls and create all visualizations
    orig_quad_points = [(pt[0], pt[1]) for pt in quad]
    orig_img_size = (img.shape[1], img.shape[0])
    detection_data = process_balls_and_visualize(img, quad, orig_quad_points, orig_img_size, model_path, model_name)
    return detection_data


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    """Main entry point for the script."""
    if len(sys.argv) != 2:
        print("Usage: python detect_table.py <image_directory>")
        sys.exit(1)

    image_dir = sys.argv[1]

    # Process all JPG images in the directory
    images = sorted(glob.glob(f"{image_dir}/*.jpg"))
    if not images:
        print(f"No JPG images found in directory: {image_dir}")
        sys.exit(1)

    print(f"Found {len(images)} images to process")
    print(f"Models to run: {MODEL_NAMES}")

    # Results dictionary: {image_name: {model_name: {stat_name: value}}}
    results = {}

    # Process each image with each model
    for image_path in images:
        image_name = Path(image_path).name
        print("=" * 100)
        print(f"Processing image: {image_name}")
        print("=" * 100)

        # Load image
        img = cv2.imread(image_path)
        if img is None:
            print(f"ERROR: Could not load image: {image_path}")
            # Store None results for all models
            results[image_name] = {}
            for model_name in MODEL_NAMES:
                results[image_name][model_name] = get_detection_stats(None)
            continue

        # Detect table quadrilateral (ONCE per image)
        quad = detect_table_and_validate(img)

        if quad is None:
            print("Table detection failed!")
            show_detection_failed(img)
            if DO_PLOT:
                cv2.waitKey(0)
            # Store None results for all models
            results[image_name] = {}
            for model_name in MODEL_NAMES:
                results[image_name][model_name] = get_detection_stats(None)
            continue

        # Show quad detection once
        # visualize_quad_detection(img, quad)
        # if DO_PLOT:
        #     cv2.waitKey(0)

        # Run ball detection with each model using the same quad
        # Store full detection data for each model
        detection_data_by_model = {}
        for model_name in MODEL_NAMES:
            detection_data = run_detect(img, quad, model_name)
            detection_data_by_model[model_name] = detection_data
            stats = get_detection_stats(detection_data)
            print(f"[{model_name}] {image_name}: {stats}")

            # Print completion banner
            print_model_banner(model_name)

        # Compute baseline comparisons and merge with base stats
        results[image_name] = {}
        baseline_data = detection_data_by_model.get('baseline')

        for model_name in MODEL_NAMES:
            model_data = detection_data_by_model[model_name]
            stats = get_detection_stats(model_data)

            # If not baseline, add comparison metrics
            if model_name != 'baseline':
                comparison_stats = compute_baseline_comparison(baseline_data, model_data)
                # Merge base stats and comparison stats
                stats.update(comparison_stats)

            results[image_name][model_name] = stats
            
    # Write all results to Excel
    write_results_to_xlsx(results)


if __name__ == "__main__":
    main()
