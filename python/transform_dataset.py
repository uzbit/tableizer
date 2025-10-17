#!/usr/bin/env python3
"""
Dataset Transformation Pipeline
Transforms pix2pockets dataset by extracting table regions and updating ball labels.

Input:  data/pix2pockets/{images,labels}/
Output: data/pix2pockets_transformed/{images,labels}/
"""

import os
import sys
import glob
import json
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

# Add utilities to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utilities import (
    extract_table_with_transformation,
    LabelRemapper,
    load_yolo_labels_simple,
)
from tableizer_ffi import detect_table_cpp


def yolo_to_pixels(labels, img_width, img_height):
    """
    Convert YOLO normalized coordinates to pixel coordinates.

    Args:
        labels: List of [class_id, center_x, center_y, width, height] (normalized)
        img_width: Image width in pixels
        img_height: Image height in pixels

    Returns:
        list: Each item is [class_id, center_x_px, center_y_px, width_px, height_px]
    """
    pixel_labels = []
    for label in labels:
        class_id, cx_norm, cy_norm, w_norm, h_norm = label

        # Convert to pixel coordinates
        cx_px = cx_norm * img_width
        cy_px = cy_norm * img_height
        w_px = w_norm * img_width
        h_px = h_norm * img_height

        pixel_labels.append([int(class_id), cx_px, cy_px, w_px, h_px])

    return pixel_labels


def pixels_to_yolo(pixel_labels, img_width, img_height):
    """
    Convert pixel coordinates to YOLO normalized format.

    Args:
        pixel_labels: List of [class_id, center_x_px, center_y_px, width_px, height_px]
        img_width: Image width in pixels
        img_height: Image height in pixels

    Returns:
        list: Each item is [class_id, center_x, center_y, width, height] (normalized 0-1)
    """
    normalized_labels = []
    for label in pixel_labels:
        class_id, cx_px, cy_px, w_px, h_px = label

        # Convert to normalized coordinates
        cx_norm = cx_px / img_width
        cy_norm = cy_px / img_height
        w_norm = w_px / img_width
        h_norm = h_px / img_height

        # Clamp to valid range [0, 1]
        cx_norm = max(0, min(1, cx_norm))
        cy_norm = max(0, min(1, cy_norm))
        w_norm = max(0, min(1, w_norm))
        h_norm = max(0, min(1, h_norm))

        normalized_labels.append([class_id, cx_norm, cy_norm, w_norm, h_norm])

    return normalized_labels


def save_yolo_labels(labels, label_path):
    """Save YOLO format labels to file."""
    os.makedirs(os.path.dirname(label_path), exist_ok=True)

    with open(label_path, "w") as f:
        for label in labels:
            class_id, cx, cy, w, h = label
            f.write(f"{int(class_id)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def transform_ball_positions(ball_centers, transformation_matrix):
    """
    Transform ball center positions using the given transformation matrix.
    """
    if len(ball_centers) == 0:
        return []

    # Convert to homogeneous coordinates
    ball_points = np.array(ball_centers, dtype=np.float32)
    ball_points_homogeneous = np.ones((len(ball_points), 3), dtype=np.float32)
    ball_points_homogeneous[:, :2] = ball_points

    # Apply transformation
    transformed_homogeneous = (transformation_matrix @ ball_points_homogeneous.T).T

    # Convert back to 2D coordinates
    transformed_points = (
        transformed_homogeneous[:, :2] / transformed_homogeneous[:, 2:3]
    )

    return transformed_points


def process_single_image(
    img_path, label_path, output_images_dir, output_labels_dir, visualize=False, require_valid=False
):
    """
    Process a single image: detect table, extract region, transform labels.

    Args:
        img_path: Path to input image
        label_path: Path to input label file
        output_images_dir: Directory to save extracted images
        output_labels_dir: Directory to save transformed labels
        visualize: Whether to show visualization
        require_valid: If True, skip images with orientation="OTHER"

    Returns:
        bool: True if successful, False if failed
    """
    try:
        # Load image
        image = cv2.imread(img_path)
        if image is None:
            print(f"Failed to load image: {img_path}")
            return False

        img_height, img_width = image.shape[:2]
        print(f"Processing: {os.path.basename(img_path)} ({img_width}x{img_height})")

        # Detect table using C++
        detection_result = detect_table_cpp(image, rotation_degrees=0)
        if not detection_result or "quad_points" not in detection_result:
            print(f"  ❌ Table detection failed")
            return False

        quad_points_list = detection_result["quad_points"]
        if len(quad_points_list) != 4:
            print(f"  ❌ Invalid quad points count: {len(quad_points_list)}")
            return False

        # Extract orientation info
        orientation = detection_result.get("orientation", "UNKNOWN")

        # Convert quad points to numpy array
        quad = np.array(
            [[pt[0], pt[1]] for pt in quad_points_list], dtype=np.float32
        )

        # Print detection info with orientation
        print(f"  ✅ Table detected - Orientation: {orientation}")

        # Check orientation if required (skip only if orientation is OTHER)
        if require_valid and orientation == "OTHER":
            print(f"  ⚠️  Skipping image with OTHER orientation")
            return False

        # Print if image is being kept
        if require_valid:
            print(f"  ✓ Keeping image")

        # Load YOLO labels using utilities function
        labels = load_yolo_labels_simple(label_path)
        if not labels:
            print(f"  ⚠️  No labels found at {label_path}")
        else:
            print(f"  📍 Found {len(labels)} labels")

        # Convert labels to pixel coordinates
        pixel_labels = yolo_to_pixels(labels, img_width, img_height)

        # Extract bounding box corners for transformation
        bbox_corners = []
        for label in pixel_labels:
            class_id, cx_px, cy_px, w_px, h_px = label
            # Calculate bbox corners: top-left, top-right, bottom-right, bottom-left
            x1 = cx_px - w_px / 2
            y1 = cy_px - h_px / 2
            x2 = cx_px + w_px / 2
            y2 = cy_px + h_px / 2
            corners = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            bbox_corners.append(corners)

        # Extract table region and get transformation matrix
        extracted_table, H_transform = extract_table_with_transformation(image, quad)

        # Transform bounding boxes
        if bbox_corners:
            transformed_labels = []
            for i, (label, corners) in enumerate(zip(pixel_labels, bbox_corners)):
                class_id = label[0]

                # Transform all 4 corners of the bounding box
                transformed_corners = transform_ball_positions(corners, H_transform)

                # Calculate new bounding box from transformed corners
                x_coords = [pt[0] for pt in transformed_corners]
                y_coords = [pt[1] for pt in transformed_corners]

                new_x1, new_x2 = min(x_coords), max(x_coords)
                new_y1, new_y2 = min(y_coords), max(y_coords)

                new_cx = (new_x1 + new_x2) / 2
                new_cy = (new_y1 + new_y2) / 2
                new_w = new_x2 - new_x1
                new_h = new_y2 - new_y1

                transformed_labels.append([class_id, new_cx, new_cy, new_w, new_h])

            pixel_labels = transformed_labels

        # Convert back to normalized YOLO format (SHOT_STUDIO_WIDTH x SHOT_STUDIO_HEIGHT)
        extracted_height, extracted_width = extracted_table.shape[:2]
        new_labels = pixels_to_yolo(pixel_labels, extracted_width, extracted_height)

        # Save extracted image
        img_basename = os.path.basename(img_path)
        output_img_path = os.path.join(output_images_dir, img_basename)
        cv2.imwrite(output_img_path, extracted_table)

        # Save transformed labels
        label_basename = os.path.basename(label_path)
        output_label_path = os.path.join(output_labels_dir, label_basename)
        save_yolo_labels(new_labels, output_label_path)

        print(f"  💾 Saved: {img_basename} + {label_basename}")

        # Show visualization while processing (if enabled)
        if visualize:
            # Show original image with quad and labels
            if bbox_corners:
                visualize_quad_and_labels(
                    image, quad, labels, f"Original: {img_basename}"
                )
            # Show transformed image with labels
            visualize_transformation(
                extracted_table, new_labels, f"Transformed: {img_basename}"
            )

        return True

    except Exception as e:
        print(f"  ❌ Error processing {img_path}: {e}")
        return False


def visualize_quad_and_labels(original_image, quad, normalized_labels, title="Original Image"):
    """Visualize the original image with detected quad and ball bounding boxes.

    Args:
        original_image: Original image with quad
        quad: Quadrilateral points (4, 2)
        normalized_labels: YOLO normalized labels
        title: Window title
    """
    overlay = original_image.copy()
    img_height, img_width = original_image.shape[:2]

    # Draw quad with red polyline and corner points (like detect_table.py)
    cv2.polylines(overlay, [quad.astype(int)], True, (0, 0, 255), 2)
    for p in quad.astype(int):
        cv2.circle(overlay, tuple(p), 6, (0, 0, 255), -1)

    # Ball colors - ordered by class: ["black", "cue", "solid", "stripe"]
    ball_colors = [
        (0, 0, 0),  # Class 0: Black
        (255, 255, 255),  # Class 1: Cue - White
        (0, 0, 255),  # Class 2: Solid - Red
        (0, 255, 255),  # Class 3: Stripe - Yellow
    ]

    # Draw ball bounding boxes
    for label in normalized_labels:
        class_id, cx_norm, cy_norm, w_norm, h_norm = label

        # Convert to pixel coordinates
        cx_px = int(cx_norm * img_width)
        cy_px = int(cy_norm * img_height)
        w_px = int(w_norm * img_width)
        h_px = int(h_norm * img_height)

        # Calculate bounding box corners
        x1 = int(cx_px - w_px / 2)
        y1 = int(cy_px - h_px / 2)
        x2 = int(cx_px + w_px / 2)
        y2 = int(cy_px + h_px / 2)

        color = ball_colors[int(class_id) % len(ball_colors)]

        # Draw bounding box rectangle
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

        # Draw center point
        cv2.circle(overlay, (cx_px, cy_px), 3, color, -1)

        # Add class label
        cv2.putText(
            overlay,
            f"{int(class_id)}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )

    cv2.imshow(title, overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def visualize_transformation(image, normalized_labels, title="Transformed Image"):
    """Visualize the transformed image with ball bounding boxes overlaid."""
    overlay = image.copy()
    img_height, img_width = image.shape[:2]

    # Ball colors - ordered by class: ["black", "cue", "solid", "stripe"]
    ball_colors = [
        (0, 0, 0),  # Class 0: Black
        (255, 255, 255),  # Class 1: Cue - White
        (0, 0, 255),  # Class 2: Solid - Red
        (0, 255, 255),  # Class 3: Stripe - Yellow
    ]

    for label in normalized_labels:
        class_id, cx_norm, cy_norm, w_norm, h_norm = label

        # Convert to pixel coordinates
        cx_px = int(cx_norm * img_width)
        cy_px = int(cy_norm * img_height)
        w_px = int(w_norm * img_width)
        h_px = int(h_norm * img_height)

        # Calculate bounding box corners
        x1 = int(cx_px - w_px / 2)
        y1 = int(cy_px - h_px / 2)
        x2 = int(cx_px + w_px / 2)
        y2 = int(cy_px + h_px / 2)

        color = ball_colors[int(class_id) % len(ball_colors)]

        # Draw bounding box rectangle
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

        # Draw center point
        cv2.circle(overlay, (cx_px, cy_px), 3, color, -1)

        # Add class label
        cv2.putText(
            overlay,
            f"{int(class_id)}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )

    cv2.imshow(title, overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def batch_process_dataset(input_dir, output_dir, visualize_samples=False, require_valid=False):
    """
    Process entire pix2pockets dataset.

    Args:
        input_dir: Path to data/pix2pockets/
        output_dir: Path to data/pix2pockets_transformed/
        visualize_samples: Show visualization for sample images
        require_valid: If True, skip images with orientation="OTHER"
    """
    # Step 1: Use LabelRemapper to create clean dataset first
    temp_dir = os.path.join(output_dir, "temp_remapped")
    oldToNewMap = {3: 3, 2: 2, 1: 1, 0: 0}  # original → remapped (from model_table.py)

    images_dir = os.path.join(input_dir, "images")
    labels_dir = os.path.join(input_dir, "labels")

    print("🔄 Step 1: Remapping labels using LabelRemapper...")
    remapper = LabelRemapper(
        src_img_dir=images_dir,
        src_lbl_dir=labels_dir,
        dst_root=temp_dir,
        old_to_new_map=oldToNewMap,
    )
    kept, dropped = remapper.run()

    # Step 2: Process the remapped data
    remapped_images_dir = os.path.join(temp_dir, "images_all")
    remapped_labels_dir = os.path.join(temp_dir, "labels_all")

    output_images_dir = os.path.join(output_dir, "images")
    output_labels_dir = os.path.join(output_dir, "labels")

    # Create output directories
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)

    # Find all remapped images
    print(f"🔄 Step 2: Processing {kept} remapped images...")
    image_patterns = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    image_files = []
    for pattern in image_patterns:
        image_files.extend(glob.glob(os.path.join(remapped_images_dir, pattern)))

    print(f"Found {len(image_files)} remapped images to transform")

    successful = 0
    failed = 0
    total = len(image_files)

    for i, img_path in enumerate(tqdm(image_files, desc="Processing images")):
        # Find corresponding label file in remapped directory
        img_name = Path(img_path).stem
        label_path = os.path.join(remapped_labels_dir, f"{img_name}.txt")

        # Process image with optional visualization and validation
        success = process_single_image(
            img_path,
            label_path,
            output_images_dir,
            output_labels_dir,
            visualize=visualize_samples,
            require_valid=require_valid,
        )

        if success:
            successful += 1
        else:
            failed += 1

    print(f"\n📊 Processing complete:")
    print(f"  ✅ Kept: {successful} / {total} images ({100*successful/total:.1f}%)")
    print(f"  ❌ Skipped: {failed} / {total} images ({100*failed/total:.1f}%)")
    print(f"  📁 Output saved to: {output_dir}")


def main():
    """Main entry point for dataset transformation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Transform pool table dataset: extract table regions and update ball labels"
    )
    parser.add_argument(
        "input_dir",
        help="Input directory containing images/ and labels/ subdirectories"
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output directory (default: <input_dir>_transformed)"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Show visualization for each processed image"
    )
    parser.add_argument(
        "--require-valid",
        action="store_true",
        help="Skip images with orientation='OTHER'"
    )

    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output if args.output else f"{input_dir}_transformed"

    if not os.path.exists(input_dir):
        print(f"❌ Input directory not found: {input_dir}")
        sys.exit(1)

    print("🏗️  Tableizer Dataset Transformation Pipeline")
    print(f"📂 Input:  {input_dir}")
    print(f"📂 Output: {output_dir}")
    if args.require_valid:
        print(f"⚙️  Filtering: Skipping images with orientation='OTHER'")
    print()

    batch_process_dataset(
        input_dir,
        output_dir,
        visualize_samples=args.visualize,
        require_valid=args.require_valid
    )


if __name__ == "__main__":
    main()
