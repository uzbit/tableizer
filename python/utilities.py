import torch
from pathlib import Path
from PIL import Image
import numpy as np
import cv2


import cv2
import numpy as np


def solid_circle(bgr_color, radius=110, size=256):
    """Return a square BGR image with a filled circle of the given colour."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    cv2.circle(img, center=(size // 2, size // 2), radius=radius,
               color=bgr_color, thickness=-1, lineType=cv2.LINE_AA)
    return img

ballImageBgrs = [
    solid_circle((0,   0, 255)),   # red   (BGR)
    solid_circle((0, 255, 255)),   # yellow
    solid_circle((255,255,255)),   # cue (white)
    solid_circle((0,   0,   0)),   # black
][::-1]

def orderQuad(pts):
    """
    Return the four 2-D points sorted **clockwise**.
    Works for any initial ordering (even a crossed one).
    """
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    c = pts.mean(axis=0)  # centroid
    angles = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])  # angle w.r.t. +x axis
    return pts[np.argsort(angles)]


def warpTable(bgrImg, quad, imagePath, outW=1000, rotate=False, scaleF=1.00):
    h0, w0 = bgrImg.shape[:2]

    # ---- 2 : 1 landscape canvas -------------------------------------------
    if rotate:
        outH = int(scaleF * 2 * outW)
    else:
        outH = int(scaleF * outW)
        outW *= 2

    dst = np.array(
        [[0, 0], [outW - 1, 0], [outW - 1, outH - 1], [0, outH - 1]], np.float32
    )

    Hpersp = cv2.getPerspectiveTransform(quad, dst)

    if not rotate:  # 2 : 1 landscape
        warp = cv2.warpPerspective(bgrImg, Hpersp, (outW, outH))
        cv2.imwrite(imagePath, warp)
        return warp, Hpersp, (outW, outH)  # <-- return size tuple too

    # ---- embed 90 ° CCW rotation ------------------------------------------
    rot = np.array([[0, 1, 0], [-1, 0, outW - 1], [0, 0, 1]], np.float32)
    Htot = rot @ Hpersp  # original → portrait canvas
    warp = cv2.warpPerspective(bgrImg, Htot, (outH, outW))
    cv2.imwrite(imagePath, warp)
    return warp, Htot


def drawBallOverlays(imgBgr, ballCenters, ballClasses, radius=12):
    vis = imgBgr.copy()

    for (x, y), cls in zip(ballCenters, ballClasses):
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


def drawShotStudio(ballCenters, ballClasses, warpImg):
    # --- load balls as BGR ---
    shotStudioTable = cv2.imread(str(Path("../data/shotstudio_table_felt_only.png")))

    print("Shotstudio:", shotStudioTable.shape)
    print("warpImg:", warpImg.shape)

    topRailWidth = 25
    # leftRail, topRail = 106, 106 - topRailWidth
    leftRail, topRail = 0, 0
    shotStudioCenters = ballCenters + np.array([leftRail, topRail])

    # Compute physical → pixel scale
    # 7-foot: 78"
    # 8-foot: 88"
    # 9-foot: 100"
    tableSize = 78
    longEdgePx = max(warpImg.shape[:2])
    ballDiaPx = int(round(longEdgePx * (2.25 / tableSize)))  # 2.25" over 100"
    ballDiaPx = max(ballDiaPx, 8)

    # Resize all ball images
    ballResized = [
        cv2.resize(b, (ballDiaPx, ballDiaPx), interpolation=cv2.INTER_AREA)
        for b in ballImageBgrs
    ]

    for center, cls in zip(shotStudioCenters, ballClasses):
        ballImg = ballResized[int(cls)]
        bh, bw = ballImg.shape[:2]
        x = int(round(center[0] - bw / 2))
        y = int(round(center[1] - bh / 2))

        # Bounds check
        if (
            x < 0
            or y < 0
            or x + bw > shotStudioTable.shape[1]
            or y + bh > shotStudioTable.shape[0]
        ):
            continue

        roi = shotStudioTable[y : y + bh, x : x + bw]

        # Create mask for non-white pixels (treating white as background)
        gray = cv2.cvtColor(ballImg, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
        maskInv = cv2.bitwise_not(mask)

        fg = cv2.bitwise_and(ballImg, ballImg, mask=mask)
        bg = cv2.bitwise_and(roi, roi, mask=maskInv)
        combined = cv2.add(bg, fg)

        shotStudioTable[y : y + bh, x : x + bw] = combined

    cv2.imshow("Shot Studio Overlay", shotStudioTable)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return shotStudioTable


def bb_IoU(bboxA, bboxB):
    """
    Parameters
    ----------
    bboxA : Array(N, 4) (xmin,ymin,w,h)
        DESCRIPTION.
    bboxB : Array(M, 4) (xmin,ymin,w,h)
        DESCRIPTION.

    Returns
    -------
    iou : Array(N,N) (0-1 if correct)

    """
    # If N or M is one
    if bboxA.ndim == 1:
        bboxA = np.array([bboxA])
    if bboxB.ndim == 1:
        bboxB = np.array([bboxB])

    N = bboxA.shape[0]
    M = bboxB.shape[0]

    xminA, yminA, wA, hA = [bboxA[:, i] for i in range(4)]
    xminB, yminB, wB, hB = [bboxB[:, i] for i in range(4)]

    iou = np.zeros((N, M))
    for i in range(N):
        area_A = wA[i] * hA[i]

        for j in range(M):
            xmin = max(xminA[i], xminB[j])
            ymin = max(yminA[i], yminB[j])
            xmax = min(xminA[i] + wA[i], xminB[j] + wB[j])
            ymax = min(yminA[i] + hA[i], yminB[j] + hB[j])

            interArea = max(0, xmax - xmin) * max(0, ymax - ymin)
            area_B = wB[j] * hB[j]

            iou[i, j] = interArea / (area_A + area_B - interArea)

    # unionArea = area_A + area_B - interArea

    # print(interArea, unionArea)
    return iou


from ultralytics import YOLO
import torch

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

    model = YOLO(model_path)        # loads v8/v9 weight in eval mode
    model.to(device)                # move once; Ultralytics handles dtype
    model.fuse()                    # optional: fuse Conv+BN for speed
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
    boxes   = results[0].boxes

    xyxy   = boxes.xyxy.cpu().numpy()      # (N,4) x1 y1 x2 y2
    confs  = boxes.conf.cpu().numpy()      # (N,)
    clsids = boxes.cls.cpu().numpy()       # (N,)

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
        iou  = bb_IoU(detection[:, :4], detection[:, :4]) - np.eye(len(detection))
        idx1, idx2 = np.where(iou >= iou_thresh)
        remove = []
        for i, j in zip(idx1, idx2):
            if detection[i, 4] <= detection[j, 4]:
                remove.append(i)
            else:
                remove.append(j)
        keep = np.setdiff1d(keep, np.unique(remove))
        detection = detection[keep]

        # -- per-class caps -------------------------------------------
        for c, n in zip([0, 1, 2, 3, 4], [7, 7, 1, 1, 18]):
            idxs = np.where(detection[:, 5] == c)[0]
            keep, drop = idxs[:n], idxs[n:]
            detection[keep, 4] = np.maximum(detection[keep, 4], conf_thresh)
            detection[drop, 4] = 0.0
        detection = detection[detection[:, 4] >= conf_thresh]
    # ----------------------------------------------------------

    img_bgr = cv2.imread(str(im_path))[:, :, ::-1]  # PIL read was RGB; caller expects BGR
    return img_bgr, detection