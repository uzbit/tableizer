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

from utilities import (
    load_detection_model,
    calculate_ball_pixel_size,
    order_quad,
)
from tableizer_ffi import detect_table_cpp, transform_points_cpp


# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

MODEL_PATH = "/Users/uzbit/Documents/projects/tableizer/tableizer/combined/weights/best.pt"
SHOTSTUDIO_BG_PATH = "../data/shotstudio_table_felt_only.png"

# Processing parameters
SHOTSTUDIO_SIZE = (840, 1680)

# Detection parameters
CONF_THRESHOLD = 0.6

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
        tuple or None: (quad_points, mask) where:
            - quad_points: numpy.ndarray of shape (4, 2) with quad coordinates
            - mask: numpy.ndarray (grayscale) or None
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
    mask = detection_result.get("mask", None)

    print(f"C++ table detection found quad (full res): {quad}")
    if mask is not None:
        print(f"Mask shape: {mask.shape}")

    return quad, mask


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

def get_balls_from_image(image_input):
    """Extract ball centers and classes from an image using YOLO detection.

    Args:
        image_input: Either a file path (str) or image array (np.ndarray)

    Returns:
        tuple: (ball_centers, ball_classes) where:
            - ball_centers: np.ndarray of shape (N, 2) with ball center coordinates
            - ball_classes: np.ndarray of shape (N,) with ball class IDs
        Returns (None, None) if no balls detected
    """
    model = load_detection_model(MODEL_PATH)

    results = model(image_input)
    boxes = results[0].boxes

    if len(boxes) == 0:
        return None, None

    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    cls_ids = boxes.cls.cpu().numpy()

    x1, y1, x2, y2 = xyxy.T
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = x2 - x1, y2 - y1

    valid_mask = confs >= CONF_THRESHOLD

    if not valid_mask.any():
        return None, None

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
        return None, None

    ball_centers = np.array([[b[0], b[1]] for b in balls])
    ball_classes = balls[:, -1]
    ball_confidences = balls[:, 4]

    print(f"Ball confidences: {ball_confidences}")
    print(f"Final ball count: {len(ball_centers)}")

    return ball_centers, ball_classes


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
    cv2.imshow("Quad via cell extremums", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


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
    cv2.imshow("Red X", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
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

    cv2.imshow(title, overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
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

def log_detection_results(image_path, ball_classes, output_file="detection_results.csv"):
    """Log detection results to CSV file.

    Args:
        image_path (str): Path to the image file
        ball_classes (np.ndarray or None): Array of ball class IDs, or None if no balls detected
        output_file (str): Output CSV file path

    Returns:
        bool: True if logging succeeded, False otherwise
    """
    try:
        # Get just the image filename
        image_name = Path(image_path).name

        # Count total balls
        num_balls = len(ball_classes) if ball_classes is not None else 0

        # Check for specific ball types (class 0 = black, class 1 = cue)
        cue_detected = False
        black_detected = False

        if ball_classes is not None and len(ball_classes) > 0:
            # Class 0 is black ball, Class 1 is cue ball
            black_detected = 0 in ball_classes
            cue_detected = 1 in ball_classes

        # Check if file exists to determine if we need to write header
        file_exists = os.path.exists(output_file)

        # Write to CSV
        with open(output_file, 'a', newline='') as f:
            writer = csv.writer(f)

            # Write header if file is new
            if not file_exists:
                writer.writerow(['image_name', 'number_balls_detected', 'cue_detected', 'black_detected'])

            # Write data row
            writer.writerow([image_name, num_balls, cue_detected, black_detected])

        print(f"Logged: {image_name} - {num_balls} balls (cue: {cue_detected}, black: {black_detected})")
        return True

    except Exception as e:
        print(f"Error logging detection results: {e}")
        return False


# ============================================================================
# MAIN PROCESSING FUNCTIONS
# ============================================================================

def process_balls_and_visualize(img, quad, orig_quad_points, orig_img_size):
    """Detect balls, transform coordinates, and create visualizations.

    Args:
        img (np.ndarray): Input image (potentially masked)
        quad (np.ndarray): Table quadrilateral points (4, 2)
        orig_quad_points (list): Original quad points as list of tuples
        orig_img_size (tuple): Original image size (width, height)

    Returns:
        np.ndarray or None: Ball class IDs array, or None if no balls detected
    """
    # Get ball detections and transform directly to ShotStudio coordinates
    ball_centers, ball_classes = get_balls_from_image(img)

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
        "Original Image with Detections",
        use_circle_indicators=True,
    )

    # Extract table region in portrait mode (same size as ShotStudio)
    extracted_table = extract_table_region(img, quad)

    # Transform ball positions for the extracted table region
    extracted_ball_positions = transform_balls_for_extracted_table(ball_centers, quad)

    # Draw balls on extracted table region (circle indicators)
    draw_ball_overlay_on_image(
        extracted_ball_positions,
        ball_classes,
        extracted_table,
        "Extracted Table with Ball Detections",
        use_circle_indicators=True,
    )

    # Get prepared ShotStudio background (loaded and cached)
    shotstudio_bg = get_shotstudio_background()
    if shotstudio_bg is None:
        return ball_classes

    # Draw balls on ShotStudio background with same positions and style as extracted table
    draw_ball_overlay_on_image(
        extracted_ball_positions,
        ball_classes,
        shotstudio_bg,
        "ShotStudio with Ball Detections",
        use_circle_indicators=True,
    )

    return ball_classes


def run_detect(image_path):
    """Run complete detection pipeline on a single image.

    Args:
        image_path (str): Path to input image file
    """
    print("-" * 100)
    print(f"Detecting for {image_path}...")

    img = cv2.imread(image_path)
    if img is None:
        print(f"ERROR: Could not load image: {image_path}")
        log_detection_results(image_path, None)
        print("-" * 100)
        return

    # Detect table quadrilateral and mask
    detection_result = detect_table_and_validate(img)

    if detection_result is not None:
        quad, mask = detection_result

        # Show quad detection
        visualize_quad_detection(img, quad)

        # Apply mask to image before ball detection if mask is available
        if mask is not None:
            # Resize mask to match original image dimensions
            mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]))
            # Apply mask - zero out non-table regions
            masked_img = cv2.bitwise_and(img, img, mask=mask_resized)
            print("Applied table mask to image for ball detection")
        else:
            masked_img = img
            print("No mask available, using original image for ball detection")

        # Check if ShotStudio background exists before processing
        if not os.path.exists(SHOTSTUDIO_BG_PATH):
            print(f"ShotStudio background not found at: {SHOTSTUDIO_BG_PATH}")
            log_detection_results(image_path, None)
            print("-" * 100)
            return

        # Process balls and create all visualizations using masked image for detection
        orig_quad_points = [(pt[0], pt[1]) for pt in quad]
        orig_img_size = (img.shape[1], img.shape[0])
        ball_classes = process_balls_and_visualize(masked_img, quad, orig_quad_points, orig_img_size)
        log_detection_results(image_path, ball_classes)
    else:
        show_detection_failed(img)
        log_detection_results(image_path, None)

    print("-" * 100)


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
    for image_path in images:
        run_detect(image_path)


if __name__ == "__main__":
    main()
