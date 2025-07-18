# billiard_table_detector.py
# ---------------------------------------------------------------
from pathlib import Path
import cv2
import glob
import random
import numpy as np
from math import atan2, cos, degrees, exp, radians, sin, sqrt
import sys
from utilities import (
    drawShotStudio,
    load_detection_model,
    get_detection,
    drawBallOverlays,
    orderQuad,
    warpTable,
)

# from auxillary.RL_usedirectly import load_RL_no_env
# from auxillary.mapping import HomographyMapping


# ──────────────────────────────────────────────────────────
#  ΔE CIEDE2000  (scalar version – good enough for one cell)
# ──────────────────────────────────────────────────────────
def deltaE2000(lab1, lab2):
    # L∈[0,100]  a,b∈[-128,127]
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2

    c1, c2 = sqrt(a1 * a1 + b1 * b1), sqrt(a2 * a2 + b2 * b2)
    cBar = (c1 + c2) / 2.0

    G = 0.5 * (1 - sqrt((cBar**7) / (cBar**7 + 25**7)))
    a1p = (1 + G) * a1
    a2p = (1 + G) * a2
    c1p, c2p = sqrt(a1p * a1p + b1 * b1), sqrt(a2p * a2p + b2 * b2)
    cBarP = (c1p + c2p) / 2.0

    h1p = (degrees(atan2(b1, a1p)) + 360) % 360
    h2p = (degrees(atan2(b2, a2p)) + 360) % 360

    dLp = L2 - L1
    dCp = c2p - c1p
    dhp = h2p - h1p
    if c1p * c2p:  # both >0
        if abs(dhp) > 180:
            dhp -= 360 * np.sign(dhp)
    else:
        dhp = 0
    dHp = 2 * sqrt(c1p * c2p) * sin(radians(dhp) / 2.0)

    LBarP = (L1 + L2) / 2.0
    hBarP = h1p + dhp / 2.0
    if abs(h1p - h2p) > 180:
        hBarP = (h1p + h2p + 360) / 2.0
    hBarP %= 360

    T = (
        1
        - 0.17 * cos(radians(hBarP - 30))
        + 0.24 * cos(radians(2 * hBarP))
        + 0.32 * cos(radians(3 * hBarP + 6))
        - 0.20 * cos(radians(4 * hBarP - 63))
    )

    Sl = 1 + (0.015 * (LBarP - 50) ** 2) / sqrt(20 + (LBarP - 50) ** 2)
    Sc = 1 + 0.045 * cBarP
    Sh = 1 + 0.015 * cBarP * T
    Rt = (
        -2
        * sqrt((cBarP**7) / (cBarP**7 + 25**7))
        * sin(radians(30 * exp(-(((hBarP - 275) / 25) ** 2))))
    )

    return sqrt(
        (dLp / Sl) ** 2
        + (dCp / Sc) ** 2
        + (dHp / Sh) ** 2
        + Rt * (dCp / Sc) * (dHp / Sh)
    )


# ──────────────────────────────────────────────────────────
#  Cellular flood-fill detector
# ──────────────────────────────────────────────────────────
class CellularTableDetector:
    """
    Implements the 5-step cellular algorithm you specified.

    Parameters
    ----------
    resizeHeight     : int   – image is scaled so height = this (keeps aspect)
    cellSize         : int   – A×A pixels per cell
    deltaEThreshold  : float – max CIEDE2000 distance to call a cell “inside”
    """

    def __init__(
        self, resizeHeight: int = 600, cellSize: int = 24, deltaEThreshold: float = 10.0
    ):
        self.resizeHeight = resizeHeight
        self.cellSize = cellSize
        self.deltaEThreshold = deltaEThreshold

    # --------------------------- public API ----------------------------------
    def detect(self, imgBgr: np.ndarray):
        """
        Returns
        -------
        maskInside : np.ndarray  uint8 mask (1 = inside felt, 0 = outside)
        debugDraw  : np.ndarray  BGR preview with inside cells outlined
        """
        small, labImg = self._prepare(imgBgr)

        rows = int(np.ceil(small.shape[0] / self.cellSize))
        cols = int(np.ceil(small.shape[1] / self.cellSize))

        visited = np.zeros((rows, cols), bool)
        inside = np.zeros((rows, cols), bool)

        # central reference colour (median Lab of the centre cell)
        centreR, centreC = rows // 2, cols // 2
        sampleDist = 20
        refList = list()
        posList = list()
        for _ in range(100):
            posx = centreR + random.randint(-sampleDist, sampleDist)
            posy = centreC + random.randint(-sampleDist, sampleDist)
            posList.append((posx, posy))
            refList.append(
                self._medianLab(
                    labImg,
                    posx,
                    posy,
                )
            )
        # for col, pos in zip(refList, posList):
        #     print(pos, col)

        refLab = np.median(np.array(refList), axis=0)

        print("TARGET COLOR", refLab)
        # BFS / flood-fill
        queue = [(centreR, centreC)]
        visited[centreR, centreC] = True
        while queue:
            r, c = queue.pop()
            cellLab = self._medianLab(labImg, r, c)
            if deltaE2000(refLab, cellLab) < self.deltaEThreshold:
                inside[r, c] = True
                # spawn 8-neighbours
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols and not visited[nr, nc]:
                            visited[nr, nc] = True
                            queue.append((nr, nc))

        mask = cv2.resize(
            inside.astype(np.uint8),  # 0/1 per cell → full res mask
            (small.shape[1], small.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

        debug = self._drawCells(small.copy(), inside)

        return small, inside, mask, debug

    def quadFromInside2(self, inside: np.ndarray) -> np.ndarray:
        """
        Given a binary grid `inside` of shape (rows, cols) and the cellSize in pixels,
        return the best-fit quadrilateral [BL, TL, BR, TR] as a 4x2 float32 array.
        """
        # 1. Get pixel coordinates of cell centers
        ys, xs = np.where(inside)
        if len(xs) == 0:
            return None

        centers = np.stack(
            [(xs + 0.5) * self.cellSize, (ys + 0.5) * self.cellSize], axis=1
        ).astype(np.float32)

        # 2. Convex hull
        hull = cv2.convexHull(centers)

        # 3. Approximate hull with a quadrilateral
        epsilon = 2.0 * cv2.arcLength(hull, True)
        approx = cv2.approxPolyN(hull, 4, epsilon, 0.2)[0]

        quad = approx.reshape(4, 2).astype(np.float32)

        # 5. Order quad as [BL, TL, BR, TR] (optional, if needed)
        return orderQuad(quad)

    def quadFromInside(self, inside: np.ndarray, width, height):
        """
        Return the 4 × 2 float32 array [BL, TL, BR, TR] taken from the most
        isolated (farthest-from-centre) inside cells in each quadrant.

        If a quadrant has no cells, it silently falls back to the previous
        “axis-extremum” logic for that corner.
        """
        ys, xs = np.where(inside)
        if xs.size == 0:
            return None

        print(width, height)
        # cell centres in pixel coords of the resized frame (x, y)
        ctrs = np.stack(
            [(xs + 0.5) * self.cellSize, (ys + 0.5) * self.cellSize], axis=1
        ).astype(np.float32)

        # -- Top-left: closest to (0, 0)
        targetTL = np.array([0, 0], dtype=np.float32)
        dist2TL = np.sum((ctrs - targetTL) ** 2, axis=1)
        idxTL = np.argmin(dist2TL)
        txl, tyl = ctrs[idxTL]

        # -- Top-right: closest to (width, 0)
        targetTR = np.array([width, -height], dtype=np.float32)
        dist2TR = np.sum((ctrs - targetTR) ** 2, axis=1)
        idxTR = np.argmin(dist2TR)
        txr, tyr = ctrs[idxTR]
        ty = min(ctrs[:, 1])
        print(idxTR, dist2TR[idxTR], txr, ctrs[idxTR])

        leftIdx = np.argmin(ctrs[:, 0])  # smallest x
        rightIdx = np.argmax(ctrs[:, 0])  # largest x
        byl = ctrs[leftIdx, 1]
        byr = ctrs[rightIdx, 1]
        bxr = ctrs[rightIdx, 0]
        bxl = ctrs[leftIdx, 0]
        by = max(ctrs[:, 1])

        topLine = self._lineFromTwoPoints((txr, tyr), (txl, tyl))
        bottomLine = self._lineFromTwoPoints((bxr, by), (bxl, by))
        leftLine = self._lineFromTwoPoints((txl, tyl), (bxl, byl))
        rightLine = self._lineFromTwoPoints((txr, tyr), (bxr, byr))

        print(topLine, bottomLine, leftLine, rightLine)

        topLeft = self._intersectLines(topLine, leftLine)
        topRight = self._intersectLines(topLine, rightLine)
        bottomLeft = self._intersectLines(bottomLine, leftLine)
        bottomRight = self._intersectLines(bottomLine, rightLine)
        # print(topLeft, topRight, bottomLeft, bottomRight)
        lines = [topLeft, topRight, bottomRight, bottomLeft]
        if not all(lines):
            return None
        quad = np.array(lines).astype(np.int16)
        # print(quad)
        print("-" * 100)
        return quad

    # --------------------------- helpers -------------------------------------
    def _intersectLines(self, line1, line2):
        A1, B1, C1 = line1
        A2, B2, C2 = line2
        D = A1 * B2 - A2 * B1
        if abs(D) < 1e-8:
            return None  # lines are parallel or coincident
        x = (B1 * C2 - B2 * C1) / D
        y = (C1 * A2 - C2 * A1) / D
        return (x, y)

    def _lineFromTwoPoints(self, p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        A = y1 - y2
        B = x2 - x1
        C = x1 * y2 - x2 * y1
        return A, B, C

    def _prepare(self, imgBgr):
        h, w = imgBgr.shape[:2]
        scale = self.resizeHeight / h
        small = cv2.resize(
            imgBgr, (int(w * scale), self.resizeHeight), interpolation=cv2.INTER_AREA
        )

        lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab[..., 0] = lab[..., 0] * 100 / 255  # 0-100
        lab[..., 1:] -= 128  # a*, b* centre at 0
        return small, lab

    def _medianLab(self, labImg, cellR, cellC):
        y1 = cellR * self.cellSize
        x1 = cellC * self.cellSize
        y2 = min(y1 + self.cellSize, labImg.shape[0])
        x2 = min(x1 + self.cellSize, labImg.shape[1])
        return np.median(labImg[y1:y2, x1:x2].reshape(-1, 3), axis=0)

    def _drawCells(self, canvas, insideMask):
        for r, c in zip(*np.where(insideMask)):
            y, x = r * self.cellSize, c * self.cellSize
            cv2.rectangle(
                canvas,
                (x, y),
                (x + self.cellSize - 1, y + self.cellSize - 1),
                (0, 255, 0),
                1,
            )
        return canvas


def main():

    if len(sys.argv) == 2:
        # Process example image (replace with actual image path)
        imageDir = sys.argv[1]
    else:
        sys.exit(1)

    image = None  # "../data/Photos-1-001/P_20250711_201047.jpg"  # "../../pix2pockets/8-Ball-Pool-3/train/images/11f_png.rf.eb0169eccfb6b264a582491457ff37b6.jpg"
    if not image:
        for image in glob.glob(f"{imageDir}/*.jpg"):
            runDetect(image)
    else:
        runDetect(image)


def runDetect(imagePath):
    print("-" * 100)
    print(f"Detecting for {imagePath}...")
    img = cv2.imread(imagePath)
    det = CellularTableDetector(
        resizeHeight=3000, cellSize=20, deltaEThreshold=20
    )  # tweak threshold ↔ lighting

    small, inside, mask, debug = det.detect(img)

    # plt.figure(figsize=(4,6)); plt.imshow(cv2.cvtColor(debug, cv2.COLOR_BGR2RGB))
    # plt.title("Inside cells (green outlines)"); plt.axis("off")
    # plt.show()

    # quad = det.quadFromInside(inside, img.shape[1], img.shape[0])
    # if quad is None:
    quad = det.quadFromInside2(inside)

    if quad is not None:
        vis = debug.copy()
        cv2.polylines(vis, [quad.astype(int)], True, (0, 0, 255), 2)
        for p in quad.astype(int):
            cv2.circle(vis, tuple(p), 6, (0, 0, 255), -1)
        print("BL, TL, BR, TR  (pixel coords in resized frame):\n", quad)
        cv2.imshow("Quad via cell extremums", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        small_path = "/tmp/small.jpg"
        cv2.imwrite(small_path, small)

        warp_path = "/tmp/warp.jpg"
        warpImg, Htot = warpTable(
            small,
            orderQuad(quad),
            warp_path,
            outW=840,  # Shotstudio: (840, 1626, 3)
            rotate=True,
        )

        warpedPts, ballClasses, warpRgb = getBalls(small_path, warpImg, Htot)
        if warpedPts is not None:
            drawShotStudio(warpedPts, ballClasses, warpRgb)

    else:
        vis = debug.copy()
        h, w = vis.shape[:2]

        # Draw from top-left to bottom-right
        cv2.line(vis, (0, 0), (w - 1, h - 1), (0, 0, 255), 2)

        # Draw from bottom-left to top-right
        cv2.line(vis, (0, h - 1), (w - 1, 0), (0, 0, 255), 2)

        cv2.imshow("Red X", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # vis = debug.copy()
        # plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        print("NO QUAD FOUND!")
    print("-" * 100)


def getBalls(origImgPath, warpImg, H):
    """
    origImgPath : original full-frame image path
    warpImg     : portrait or landscape cloth image from warpTable
    H           : homography original → warpImg (already includes rotation)
    """
    modelPath = "/Users/uzbit/Documents/projects/pix2pockets/detection_model_weight/detection_model.pt"
    model = load_detection_model(modelPath)

    # ---------- run detector on ORIGINAL frame ------------------------------
    origBgr, dets = get_detection(
        origImgPath, model, post_process=True, conf_thresh=0.01, iou_thresh=0.5
    )

    # keep only balls
    balls = dets[dets[:, 5] != 4]
    if balls.size == 0:
        print("No balls.")
        return None, None, None

    # ---------- centres in original-pixel space -----------------------------
    ballCenters = np.array([[b[0], b[1]] for b in balls])
    ballClasses = balls[:, -1]

    # ---------- project into warp space -------------------------------------
    pts = np.asarray(ballCenters, np.float32).reshape(-1, 1, 2)
    warpedPts = cv2.perspectiveTransform(pts, H).reshape(-1, 2)

    print(ballCenters)
    print(warpedPts)

    # ---------- draw on the *warp* image (RGB) ------------------------------
    warpRgb = cv2.cvtColor(warpImg, cv2.COLOR_BGR2RGB)
    overlay = drawBallOverlays(warpRgb, warpedPts, ballClasses, radius=14)

    cv2.imshow("Balls on warp", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return warpedPts, ballClasses, warpRgb


if __name__ == "__main__":
    main()
    # getBalls("warp90.jpg")
