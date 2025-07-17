import torch
import os
from PIL import Image
import numpy as np
import cv2
from skimage import transform
from matplotlib import pyplot as plt


from pdf2image import convert_from_path   # uses Poppler
import cv2
import numpy as np


def pdfPageToCv2(pdfPath, page=0, dpi=300):
    """Return a BGR image (OpenCV) of the requested page."""
    pilPages = convert_from_path(pdfPath, dpi=dpi, first_page=page+1, last_page=page+1)
    if not pilPages:
        raise ValueError("No such page in PDF")
    pilImg   = pilPages[0]                            # PIL.Image
    rgb      = np.array(pilImg)                       # RGB H×W×3 uint8
    bgr      = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)   # OpenCV prefers BGR
    return bgr

def getBallInfo(ball_data, H):
    # print(H)
    # tform = transform.ProjectiveTransform(H)
    # print(tform)
    ball_centers = np.array([[b[0], b[1]] for b in ball_data])
    ball_classes = ball_data[:, -1]

    # ball_H = tform(ball_centers)
    # print(ball_H)
    print("*" * 50 + "ORIG BALL INFO" + "*" * 50)
    print(ball_centers)
    print(ball_classes)
    print("*" * 50 + "ORIG BALL INFO" + "*" * 50)
    return ball_centers, ball_classes


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


def load_detection_model(model_path):
    model = torch.hub.load(
        "ultralytics/yolov5", "custom", path=model_path
    )  # local model
    model.eval()

    # torchhub messes with matplotlib
    # %matplotlib inline
    return model


def get_detection(
    im_path,
    model,
    post_process=True,
    conf_thresh=0.4,
    iou_thresh=0.5,
    output_format="center",
):
    im = Image.open(im_path)
    results = model(im)

    detection = results.pandas().xyxy[0].to_numpy()

    xmin = detection[:, 0]
    ymin = detection[:, 1]
    xmax = detection[:, 2]
    ymax = detection[:, 3]
    w = xmax - xmin
    h = ymax - ymin

    if output_format == "corner":
        transformed_boxes = np.array([xmin, ymin, w, h]).T
    elif output_format == "center":
        x_center = xmin + w / 2
        y_center = ymin + h / 2
        transformed_boxes = np.array([x_center, y_center, w, h]).T
    else:
        print(
            f"output_format must either be 'corner' or 'center'. Your input was {output_format}."
        )
    detection[:, :4] = transformed_boxes

    detection = detection[:, :-1].astype(np.float64)

    if post_process:
        detection = detection[
            detection[:, 4].argsort()[::-1]
        ]  # Sort according to confidence

        # Run non-max-suppresion
        keep_list = np.arange(0, detection.shape[0])
        remove_list = []
        iou = bb_IoU(detection[:, :4], detection[:, :4]) - np.eye(detection.shape[0])

        overlaps = np.where((iou >= iou_thresh))
        overlaps = np.stack(overlaps, 1)
        for idx1, idx2 in overlaps:
            conf1 = detection[idx1, 4]
            conf2 = detection[idx2, 4]
            if conf1 <= conf2:
                remove_list.append(idx1)
            else:
                remove_list.append(idx2)

        remove_list = np.unique(remove_list)
        keep_list = [k for k in keep_list if k not in remove_list]

        detection = detection[keep_list]

        # Keep 7 highest striped and solids 1 cue and black and 18 dots
        for c, n in zip([0, 1, 2, 3, 4], [7, 7, 1, 1, 18]):
            idxs = [i for i, x in enumerate(detection[:, 5]) if x == c]
            keep = idxs[:n]
            remove = idxs[n:]

            for i in keep:
                detection[i, 4] = max(
                    detection[i, 4], conf_thresh
                )  # We are at least 'conf_thresh' confident about these predictions

            for j in remove:
                detection[j, 4] = 0.0

        # Remove detections with conf_score less than threshhold
        detection = detection[detection[:, 4] >= conf_thresh]

    return cv2.imread(str(im_path))[:, :, ::-1], detection
