#!/usr/bin/env python3
"""
model_table.py – Build a *four-class* YOLOv5 model for
(pool) stripes, solids, cue ball, and black ball.

Author: ChatGPT (o3)
"""

from __future__ import annotations
import argparse, shutil, random, subprocess, sys, json
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────
# 1.  Label cleaning & remapping
# ──────────────────────────────────────────────────────────
class LabelRemapper:
    """
    Drops unwanted classes and remaps the remaining ones to {0,1,2,3}
    """

    def __init__(
        self,
        srcImgDir: Path,
        srcLblDir: Path,
        dstRoot: Path,
        oldToNewMap: Dict[int, int],
        imgExts: Tuple[str, ...] = (".jpg", ".jpeg", ".png"),
    ) -> None:
        self.srcImgDir = srcImgDir
        self.srcLblDir = srcLblDir
        self.dstRoot = dstRoot
        self.oldToNewMap = oldToNewMap
        self.imgExts = imgExts
        self.dstImgDir = dstRoot / "images_all"
        self.dstLblDir = dstRoot / "labels_all"
        self.dstImgDir.mkdir(parents=True, exist_ok=True)
        self.dstLblDir.mkdir(parents=True, exist_ok=True)

    def _remap_file(self, lblPath: Path, imgPath: Path) -> bool:
        """
        Returns True if at least one line survived the filter.
        """
        outLines: List[str] = []
        with lblPath.open() as fp:
            for line in fp:
                if not line.strip():
                    continue
                oldId, *rest = line.strip().split()
                oldId = int(oldId)
                if oldId not in self.oldToNewMap:
                    # unwanted (e.g., diamonds) → skip
                    continue
                newId = self.oldToNewMap[oldId]
                outLines.append(" ".join([str(newId), *rest]))
        if not outLines:
            return False  # no valid objects, skip this image

        # copy image + write new label
        newImg = self.dstImgDir / imgPath.name
        newLbl = self.dstLblDir / lblPath.name
        shutil.copy2(imgPath, newImg)
        newLbl.write_text("\n".join(outLines) + "\n")
        return True

    def run(self) -> None:
        labelFiles = sorted(self.srcLblDir.glob("*.txt"))
        dropped, kept = 0, 0
        for lbl in labelFiles:
            imgName = lbl.stem + ".jpg"  # default
            for ext in self.imgExts:
                candidate = self.srcImgDir / (lbl.stem + ext)
                if candidate.exists():
                    imgName = candidate.name
                    break
            imgPath = self.srcImgDir / imgName
            if not imgPath.exists():
                print(f"[WARN] image missing for {lbl}", file=sys.stderr)
                continue
            if self._remap_file(lbl, imgPath):
                kept += 1
            else:
                dropped += 1
        print(f"[LabelRemapper] kept {kept} imgs, dropped {dropped} (no target classes)")


# ──────────────────────────────────────────────────────────
# 2.  Train/val/test split
# ──────────────────────────────────────────────────────────
class DataSplitter:
    """
    Stratified split so each subset has roughly the same class distribution.
    """

    def __init__(
        self,
        allImgDir: Path,
        allLblDir: Path,
        dstRoot: Path,
        split: Tuple[float, float, float] = (0.8, 0.15, 0.05),
        seed: int = 42,
    ) -> None:
        self.allImgDir = allImgDir
        self.allLblDir = allLblDir
        self.dstRoot = dstRoot
        self.split = split
        random.seed(seed)

    def _collect_by_major_class(self) -> Dict[int, List[str]]:
        """
        Groups image filenames by *first* class appearing in its label.
        Simple but works fine for stratification.
        """
        buckets = defaultdict(list)
        for lblPath in self.allLblDir.glob("*.txt"):
            firstCls = int(lblPath.read_text().split()[0])
            buckets[firstCls].append(lblPath.stem)
        return buckets

    def _write_subset(self, subset: str, stems: List[str]) -> None:
        imgDst = self.dstRoot / "images" / subset
        lblDst = self.dstRoot / "labels" / subset
        imgDst.mkdir(parents=True, exist_ok=True)
        lblDst.mkdir(parents=True, exist_ok=True)
        for stem in stems:
            shutil.move(str(self.allImgDir / f"{stem}.jpg"), imgDst / f"{stem}.jpg")
            shutil.move(str(self.allLblDir / f"{stem}.txt"), lblDst / f"{stem}.txt")

    def run(self) -> None:
        buckets = self._collect_by_major_class()
        train, val, test = [], [], []
        for stems in buckets.values():
            random.shuffle(stems)
            n = len(stems)
            nTrain = int(self.split[0] * n)
            nVal   = int(self.split[1] * n)
            train += stems[:nTrain]
            val   += stems[nTrain:nTrain + nVal]
            test  += stems[nTrain + nVal :]
        self._write_subset("train", train)
        self._write_subset("val",   val)
        self._write_subset("test",  test)
        print(f"[DataSplitter] moved {len(train)} train, {len(val)} val, {len(test)} test images")


# ──────────────────────────────────────────────────────────
# 3.  data.yaml writer
# ──────────────────────────────────────────────────────────
class YamlWriter:
    def __init__(self, datasetRoot: Path, names: List[str]) -> None:
        self.datasetRoot = datasetRoot
        self.names = names

    def write(self) -> Path:
        content = {
            "path": str(self.datasetRoot),
            "train": "images/train",
            "val":   "images/val",
            "test":  "images/test",
            "nc": len(self.names),
            "names": self.names,
        }
        yamlPath = self.datasetRoot / "data.yaml"
        yamlPath.write_text(json.dumps(content, indent=2).replace('"', ""))
        print(f"[YamlWriter] wrote {yamlPath}")
        return yamlPath


# ──────────────────────────────────────────────────────────
# 4.  Training wrapper
# ──────────────────────────────────────────────────────────
class Yolo5Trainer:
    """
    Thin wrapper around Ultralytics train.py
    """

    def __init__(
        self,
        repoDir: Path,
        dataYaml: Path,
        weights: str = "yolov5s.pt",
        epochs: int = 150,
        imgSize: int = 640,
        batch: int = 16,
        project: str = "tableizer",
        name: str = "exp",
    ) -> None:
        self.repoDir = repoDir
        self.dataYaml = dataYaml
        self.weights = weights
        self.epochs = epochs
        self.imgSize = imgSize
        self.batch = batch
        self.project = project
        self.name = name

    def _clone_repo_if_needed(self) -> None:
        if (self.repoDir / "train.py").exists():
            return
        print("[Yolo5Trainer] cloning Ultralytics YOLOv5 …")
        subprocess.run(
            ["git", "clone", "https://github.com/ultralytics/yolov5", str(self.repoDir)],
            check=True,
        )
        subprocess.run([sys.executable, "-m", "pip", "install", "-qr", "requirements.txt"],
                       cwd=self.repoDir, check=True)

    def train(self) -> None:
        self._clone_repo_if_needed()
        cmd = [
            sys.executable, "train.py",
            "--img", str(self.imgSize),
            "--batch", str(self.batch),
            "--epochs", str(self.epochs),
            "--data", str(self.dataYaml),
            "--weights", self.weights,
            "--device", "mps",
            "--workers", "40",
            "--project", self.project,
            "--name", self.name,
            "--cache"
        ]
        print("[Yolo5Trainer] launching:", " ".join(cmd))
        subprocess.run(cmd, cwd=self.repoDir, check=True)

# ──────────────────────────────────────────────────────────
# 5.  Orchestration / CLI
# ──────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Build 4-class YOLOv5 pool model")
    parser.add_argument("--config", default=None,
                        help="Optional JSON file overriding the CONFIG dict")
    args = parser.parse_args()

    # Default config (override with --config FILE.json)
    CONFIG = {
        "srcImgDir": "data/pix2pocket/images",
        "srcLblDir": "data/pix2pocket/labels",
        "dstRoot":   "/tmp/workdir",
        # keys = original IDs, vals = new IDs 0-3
        # Adjust to match *your* original label scheme!
        "oldToNewMap": {
            4: 3, #9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0,   # Stripes 
            3: 2, #1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1,         # Solids  
            1: 1,                                             # Cue     
            0: 0,                                             # Black   
        },
        "classNames": ["stripe", "solid", "cue", "black"],
        "split": [0.8, 0.15, 0.05],
        "trainer": {
            "repoDir": "yolov5",
            "weights": "yolov5s.pt",
            "epochs": 100,
            "imgSize": 640,
            "batch": 20,
            "project": "tableizer",
            "name": "8ball",
        }
    }

    if args.config:
        CONFIG.update(json.loads(Path(args.config).read_text()))

    # Paths
    srcImgDir = Path(CONFIG["srcImgDir"]).expanduser()
    srcLblDir = Path(CONFIG["srcLblDir"]).expanduser()
    dstRoot   = Path(CONFIG["dstRoot"]).expanduser()
    dstRoot.mkdir(exist_ok=True)

    # 1. remap labels
    LabelRemapper(
        srcImgDir=srcImgDir,
        srcLblDir=srcLblDir,
        dstRoot=dstRoot,
        oldToNewMap=CONFIG["oldToNewMap"],
    ).run()

    # 2. split
    DataSplitter(
        allImgDir=dstRoot / "images_all",
        allLblDir=dstRoot / "labels_all",
        dstRoot=dstRoot,
        split=tuple(CONFIG["split"]),
    ).run()

    # 3. yaml
    dataYaml = YamlWriter(dstRoot, CONFIG["classNames"]).write()

    # 4. train
    trainerCfg = CONFIG["trainer"]
    Yolo5Trainer(
        repoDir=Path(trainerCfg["repoDir"]),
        dataYaml=dataYaml,
        weights=trainerCfg["weights"],
        epochs=trainerCfg["epochs"],
        imgSize=trainerCfg["imgSize"],
        batch=trainerCfg["batch"],
        project=trainerCfg["project"],
        name=trainerCfg["name"],
    ).train()

    # 5. plot
    runDir = Path(trainerCfg["repoDir"]) / trainerCfg["project"] / trainerCfg["name"]
    MetricsPlotter(runDir).plot()

if __name__ == "__main__":
    main()