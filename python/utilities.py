import torch
from pathlib import Path
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO
import torch
import os

import cv2
import numpy as np


def calculate_ball_pixel_size(image, table_size=78):
    """
    Calculate ball pixel size based on table dimensions.

    Args:
        image: Input image to get pixel dimensions from
        table_size: Table size in inches (78" for 7-foot, 88" for 8-foot, 100" for 9-foot)

    Returns:
        int: Ball diameter in pixels
    """
    BALL_DIAMETER_INCHES = 2.25
    long_edge_px = max(image.shape[:2])
    return int(round(long_edge_px * (BALL_DIAMETER_INCHES / table_size)))


def solid_circle(bgr_color, radius=110, size=256):
    """Return a square BGR image with a filled circle of the given colour."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    cv2.circle(
        img,
        center=(size // 2, size // 2),
        radius=radius,
        color=bgr_color,
        thickness=-1,
        lineType=cv2.LINE_AA,
    )
    return img


ball_image_bgrs = [
    solid_circle((0, 0, 255)),  # red   (BGR)
    solid_circle((0, 255, 255)),  # yellow
    solid_circle((255, 255, 255)),  # cue (white)
    solid_circle((0, 0, 0)),  # black
][::-1]


def order_quad(pts):
    """
    Return the four 2-D points sorted **clockwise**.
    Works for any initial ordering (even a crossed one).
    """
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    c = pts.mean(axis=0)  # centroid
    angles = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])  # angle w.r.t. +x axis
    return pts[np.argsort(angles)]


def warp_table(bgr_img, quad, image_path, out_w=864, out_h=1680):
    """
    Warp table to ShotStudio dimensions (864x1680) with no rotation.
    Simple perspective transform to match app/assets/images/shotstudio_table_felt_only.png
    """
    dst = np.array(
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], np.float32
    )

    H = cv2.getPerspectiveTransform(quad, dst)
    warp = cv2.warpPerspective(bgr_img, H, (out_w, out_h))
    cv2.imwrite(image_path, warp)
    return warp, H


def draw_ball_overlays(img_bgr, ball_centers, ball_classes, radius=12):
    vis = img_bgr.copy()

    for (x, y), cls in zip(ball_centers, ball_classes):
        color = (
            (0, 255, 0) if cls == 1 else (0, 0, 255)
        )  # Green for class 1, Red for class 0
        center = (int(round(x)), int(round(y)))
        cv2.circle(vis, center, radius, color, 2)  # thickness 2

        # Optional: label
        cv2.putText(
            vis,
            f"{cls}",
            (center[0] + 8, center[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    return vis


def draw_shot_studio(ball_centers, ball_classes, warp_img):
    shot_studio_table = cv2.imread(str(Path("../data/shotstudio_table_felt_only.png")))

    print("Shotstudio:", shot_studio_table.shape)
    print("warp_img:", warp_img.shape)

    top_rail_width = 25
    left_rail, top_rail = 0, 0
    shot_studio_centers = ball_centers + np.array([left_rail, top_rail])

    ball_dia_px = calculate_ball_pixel_size(warp_img, table_size=78)
    ball_dia_px = max(ball_dia_px, 8)

    ball_resized = [
        cv2.resize(b, (ball_dia_px, ball_dia_px), interpolation=cv2.INTER_AREA)
        for b in ball_image_bgrs
    ]

    for center, cls in zip(shot_studio_centers, ball_classes):
        ball_img = ball_resized[int(cls)]
        bh, bw = ball_img.shape[:2]
        x = int(round(center[0] - bw / 2))
        y = int(round(center[1] - bh / 2))

        if (
            x < 0
            or y < 0
            or x + bw > shot_studio_table.shape[1]
            or y + bh > shot_studio_table.shape[0]
        ):
            continue

        roi = shot_studio_table[y : y + bh, x : x + bw]

        gray = cv2.cvtColor(ball_img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
        mask_inv = cv2.bitwise_not(mask)

        fg = cv2.bitwise_and(ball_img, ball_img, mask=mask)
        bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
        combined = cv2.add(bg, fg)

        shot_studio_table[y : y + bh, x : x + bw] = combined

    cv2.imshow("Shot Studio Overlay", shot_studio_table)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return shot_studio_table


def bb_IoU(bbox_a, bbox_b):
    """
    Parameters
    ----------
    bbox_a : Array(N, 4) (xmin,ymin,w,h)
        DESCRIPTION.
    bbox_b : Array(M, 4) (xmin,ymin,w,h)
        DESCRIPTION.

    Returns
    -------
    iou : Array(N,N) (0-1 if correct)

    """
    # If N or M is one
    if bbox_a.ndim == 1:
        bbox_a = np.array([bbox_a])
    if bbox_b.ndim == 1:
        bbox_b = np.array([bbox_b])

    N = bbox_a.shape[0]
    M = bbox_b.shape[0]

    xmin_a, ymin_a, w_a, h_a = [bbox_a[:, i] for i in range(4)]
    xmin_b, ymin_b, w_b, h_b = [bbox_b[:, i] for i in range(4)]

    iou = np.zeros((N, M))
    for i in range(N):
        area_a = w_a[i] * h_a[i]

        for j in range(M):
            xmin = max(xmin_a[i], xmin_b[j])
            ymin = max(ymin_a[i], ymin_b[j])
            xmax = min(xmin_a[i] + w_a[i], xmin_b[j] + w_b[j])
            ymax = min(ymin_a[i] + h_a[i], ymin_b[j] + h_b[j])

            inter_area = max(0, xmax - xmin) * max(0, ymax - ymin)
            area_b = w_b[j] * h_b[j]

            iou[i, j] = inter_area / (area_a + area_b - inter_area)

    # union_area = area_a + area_b - inter_area

    # print(interArea, unionArea)
    return iou


def load_detection_model(model_path: str, device: str = None):
    """
    Load a YOLO-v8/9 .pt file (or .pt/.onnx/.torchscript after export).

    Parameters
    ----------
    model_path : str
        Path to your trained weight file, e.g. "runs/detect/exp/weights/best.pt".
    device : str, optional
        "cpu", "cuda:0", "mps", …  If ``None`` it follows torch.cuda.is_available().

    Returns
    -------
    YOLO
        An Ultralytics YOLO object ready for inference:  results = model(img)
    """
    # Pick a sensible default device
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = YOLO(model_path)  # loads v8/v9 weight in eval mode
    model.to(device)  # move once; Ultralytics handles dtype
    model.fuse()  # optional: fuse Conv+BN for speed
    return model


def get_detection(
    im_path: str,
    model: YOLO,
    *,
    post_process: bool = True,
    conf_thresh: float = 0.4,
    iou_thresh: float = 0.5,
    output_format: str = "center",  # 'center' or 'corner'
):
    """
    Run YOLO-v8/v9 inference on one image and return (BGR image, detections).

    Parameters
    ----------
    im_path : str
        Path to the input picture.
    model : ultralytics.YOLO
        A YOLO object in eval mode.
    post_process : bool, default True
        Whether to run the custom NMS / per-class capping logic.
    conf_thresh : float
        Minimum confidence to keep a detection (after post-process).
    iou_thresh : float
        IoU threshold for the hand-rolled NMS.
    output_format : {'center', 'corner'}
        Whether the first four numbers are (cx,cy,w,h) or (x,y,w,h).

    Returns
    -------
    img_bgr : np.ndarray  (H, W, 3)  uint8
        The original image loaded with cv2 (BGR order).
    detection : np.ndarray  (N, 6)  float64
        Columns: x/ cx, y/ cy, w, h, confidence, class_id
    """

    # -------------------- 1. run inference --------------------
    im = Image.open(im_path)
    results = model(im)
    boxes = results[0].boxes

    xyxy = boxes.xyxy.cpu().numpy()  # (N,4) x1 y1 x2 y2
    confs = boxes.conf.cpu().numpy()  # (N,)
    clsids = boxes.cls.cpu().numpy()  # (N,)

    # convert xyxy → (x, y, w, h) in corner coords
    x1, y1, x2, y2 = xyxy.T
    w, h = x2 - x1, y2 - y1

    if output_format == "corner":
        transformed = np.stack([x1, y1, w, h], axis=1)
    elif output_format == "center":
        cx, cy = x1 + w / 2, y1 + h / 2
        transformed = np.stack([cx, cy, w, h], axis=1)
    else:
        raise ValueError("output_format must be 'corner' or 'center'.")

    # final array: [coords, conf, cls]
    detection = np.concatenate(
        [transformed, confs[:, None], clsids[:, None]], axis=1
    ).astype(np.float64)

    # -------------------- 2. optional post-process --------------------
    if post_process and detection.size:
        # sort by confidence desc
        detection = detection[detection[:, 4].argsort()[::-1]]

        # -- naive NMS (same as original helper) -----------------------
        keep = np.arange(len(detection))
        iou = bb_IoU(detection[:, :4], detection[:, :4]) - np.eye(len(detection))
        idx1, idx2 = np.where(iou >= iou_thresh)
        remove = []
        for i, j in zip(idx1, idx2):
            if detection[i, 4] <= detection[j, 4]:
                remove.append(i)
            else:
                remove.append(j)
        keep = np.setdiff1d(keep, np.unique(remove))
        detection = detection[keep]

        # -- per-class caps removed -------------------------------------------
        # Let all balls through without artificial limits
        detection = detection[detection[:, 4] >= conf_thresh]
    # ----------------------------------------------------------

    img_bgr = cv2.imread(str(im_path))[
        :, :, ::-1
    ]  # PIL read was RGB; caller expects BGR
    return img_bgr, detection


def extract_table_with_transformation(image, quad):
    """
    Extract table region and return both image and transformation matrix.
    Always outputs portrait mode (864x1680) like ShotStudio.

    Args:
        image: Input image (BGR format)
        quad: 4 corner points of detected table

    Returns:
        tuple: (extracted_image, transformation_matrix)
    """
    # Order the quad points properly
    ordered_quad = order_quad(quad)

    # Check if rotation is needed based on quad dimensions
    top_length = np.linalg.norm(ordered_quad[1] - ordered_quad[0])
    right_length = np.linalg.norm(ordered_quad[2] - ordered_quad[1])
    needs_rotation = top_length > right_length * 1.75

    # ShotStudio dimensions (always portrait: SHOT_STUDIO_WIDTH x SHOT_STUDIO_HEIGHT)
    out_w, out_h = 864, 1680

    if needs_rotation:
        # Table is landscape in image, needs rotation to portrait
        landscape_dst = np.array(
            [[0, 0], [out_h - 1, 0], [out_h - 1, out_w - 1], [0, out_w - 1]],
            dtype=np.float32,
        )

        h_landscape = cv2.getPerspectiveTransform(ordered_quad, landscape_dst)

        # Add ROTATION_DEGREES_CCW° CCW rotation: landscape (SHOT_STUDIO_HEIGHT x SHOT_STUDIO_WIDTH) -> portrait (SHOT_STUDIO_WIDTH x SHOT_STUDIO_HEIGHT)
        rot = np.array([[0, 1, 0], [-1, 0, out_h - 1], [0, 0, 1]], np.float32)
        h_final = rot @ h_landscape
        extracted = cv2.warpPerspective(image, h_final, (out_w, out_h))
    else:
        # Table is already portrait in image, direct warp
        portrait_dst = np.array(
            [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
            dtype=np.float32,
        )

        h_final = cv2.getPerspectiveTransform(ordered_quad, portrait_dst)
        extracted = cv2.warpPerspective(image, h_final, (out_w, out_h))

    return extracted, h_final


class LabelRemapper:
    """
    Drops unwanted classes and remaps the remaining ones to {0,1,2,3}
    Moved from model_table.py for reuse in dataset transformation.
    """

    def __init__(
        self,
        src_img_dir,
        src_lbl_dir,
        dst_root,
        old_to_new_map,
        img_exts=(".jpg", ".jpeg", ".png"),
    ):
        self.src_img_dir = src_img_dir
        self.src_lbl_dir = src_lbl_dir
        self.dst_root = dst_root
        self.old_to_new_map = old_to_new_map
        self.img_exts = img_exts

        # Create directories
        self.dst_img_dir = os.path.join(dst_root, "images_all")
        self.dst_lbl_dir = os.path.join(dst_root, "labels_all")
        os.makedirs(self.dst_img_dir, exist_ok=True)
        os.makedirs(self.dst_lbl_dir, exist_ok=True)

    def _remap_file(self, lbl_path, img_path):
        """Remap labels in a single file."""
        out_lines = []

        with open(lbl_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if not parts:
                    continue

                old_id = int(parts[0])
                rest = parts[1:]  # coordinates and any extra info

                if old_id not in self.old_to_new_map:  # e.g. diamonds
                    continue

                new_id = self.old_to_new_map[old_id]
                out_lines.append(" ".join([str(new_id)] + rest))

        if not out_lines:
            return False  # skip images with no target objects

        # Copy image and save remapped labels
        import shutil

        shutil.copy2(
            img_path, os.path.join(self.dst_img_dir, os.path.basename(img_path))
        )

        with open(os.path.join(self.dst_lbl_dir, os.path.basename(lbl_path)), "w") as f:
            f.write("\n".join(out_lines) + "\n")

        return True

    def run(self):
        """Process all label files."""
        import glob

        kept = dropped = 0
        label_files = glob.glob(os.path.join(self.src_lbl_dir, "*.txt"))

        for lbl_path in sorted(label_files):
            lbl_name = os.path.splitext(os.path.basename(lbl_path))[0]

            # Find corresponding image
            img_path = None
            for ext in self.img_exts:
                potential_img = os.path.join(self.src_img_dir, f"{lbl_name}{ext}")
                if os.path.exists(potential_img):
                    img_path = potential_img
                    break

            if img_path is None:
                print(f"[WARN] missing image for {lbl_path}")
                continue

            # remap_file returns True if the label survives
            if self._remap_file(lbl_path, img_path):
                kept += 1  # count this image as kept
            else:
                dropped += 1  # label had no target classes

        print(f"[LabelRemapper] kept {kept} imgs, dropped {dropped}")
        return kept, dropped


def load_yolo_labels_simple(label_path):
    """
    Simple YOLO label loader that preserves original format.
    Returns labels as [class_id, center_x, center_y, width, height]
    """
    labels = []
    if not os.path.exists(label_path):
        return labels

    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) >= 5:
                    # Take first 5 values: class_id, cx, cy, w, h
                    class_id = int(parts[0])
                    coords = list(map(float, parts[1:5]))
                    labels.append([class_id] + coords)

    return labels
