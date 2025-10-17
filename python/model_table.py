#!/usr/bin/env python3
"""
model_table.py – Train a *four-class* pool-ball detector
(stripe, solid, cue, black) with the modern Ultralytics YOLO API
(v8, v9, v10 … anything the wheel ships).

Author: ChatGPT (o3)
"""

from __future__ import annotations
import argparse, shutil, random, json, sys
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import warnings
import yaml

warnings.filterwarnings("ignore")

# EXPORT TORCHSCRIPT WITH:
# yolo export model=/Users/uzbit/Documents/projects/tableizer/tableizer/7/weights/best.pt format=onnx device=cpu imgsz=800 simplify=True dynamic=False
# EXPORT ONNX WITH:
# yolo export model=tableizer/expN/weights/best.pt format=onnx device=cpu imgsz=1280 simplify=True dynamic=False opset=17 half=False
# cp tableizer/expN/weights/best.onnx ./app/assets/detection_model.onnx

# ──────────────────────────────────────────────────────────
# 0.  Optional: ensure ultralytics is installed
# ──────────────────────────────────────────────────────────
try:
    from ultralytics import YOLO
except ImportError:
    sys.exit(
        "❌  Ultralytics package missing. "
        "Activate the venv and run: pip install -U ultralytics"
    )

# ──────────────────────────────────────────────────────────
# 1.  Import label remapper from utilities
# ──────────────────────────────────────────────────────────
from utilities import LabelRemapper


# ──────────────────────────────────────────────────────────
# 2.  Train/val/test split
# ──────────────────────────────────────────────────────────
class DataSplitter:
    """Stratified split so each subset has ≈ same class histogram."""

    def __init__(
        self,
        allImgDir: Path,
        allLblDir: Path,
        dstRoot: Path,
        split=(0.8, 0.15, 0.05),
        seed=42,
    ):
        self.allImgDir, self.allLblDir = allImgDir, allLblDir
        self.dstRoot, self.split = dstRoot, split
        random.seed(seed)

    def _collect(self) -> Dict[int, List[str]]:
        buckets = defaultdict(list)
        for lbl in self.allLblDir.glob("*.txt"):
            cls0 = int(lbl.read_text().split()[0])
            buckets[cls0].append(lbl.stem)
        return buckets

    def _dump(self, subset: str, stems: List[str]):
        (self.dstRoot / f"images/{subset}").mkdir(parents=True, exist_ok=True)
        (self.dstRoot / f"labels/{subset}").mkdir(parents=True, exist_ok=True)
        for s in stems:
            shutil.move(
                str(self.allImgDir / f"{s}.jpg"),
                self.dstRoot / f"images/{subset}/{s}.jpg",
            )
            shutil.move(
                str(self.allLblDir / f"{s}.txt"),
                self.dstRoot / f"labels/{subset}/{s}.txt",
            )

    def run(self):
        train, val, test = [], [], []
        for stems in self._collect().values():
            random.shuffle(stems)
            n = len(stems)
            nTr, nVal = int(self.split[0] * n), int(self.split[1] * n)
            train += stems[:nTr]
            val += stems[nTr : nTr + nVal]
            test += stems[nTr + nVal :]
        self._dump("train", train)
        self._dump("val", val)
        self._dump("test", test)
        print(f"[DataSplitter] train:{len(train)}  val:{len(val)}  test:{len(test)}")


# ──────────────────────────────────────────────────────────
# 3.  data.yaml writer
# ──────────────────────────────────────────────────────────
class YamlWriter:
    def __init__(self, root: Path, names: List[str]):
        self.root, self.names = root, names

    def write(self) -> Path:
        path = self.root / "data.yaml"
        yaml = {
            "path": str(self.root),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "nc": len(self.names),
            "names": self.names,
        }
        path.write_text(json.dumps(yaml, indent=2).replace('"', ""))
        print(f"[YamlWriter] wrote {path}")
        return path


# ──────────────────────────────────────────────────────────
# 4.  Ultralytics trainer wrapper
# ──────────────────────────────────────────────────────────
class UltraTrainer:
    """Thin wrapper around `ultralytics.YOLO(...).train()`."""

    def __init__(
        self,
        model: str,
        dataYaml: Path,
        hypYaml: str | None,
        *,
        epochs=150,
        imgsz=1280,
        batch=16,
        device="mps",  # "cpu", "mps", "0", "0,1"
        workers=40,
        project="tableizer",
        name="exp",
        cache=True,
    ):
        self.model, self.dataYaml, self.hypYaml = model, dataYaml, hypYaml
        self.kw = dict(
            data=str(dataYaml),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            workers=workers,
            project=project,
            name=name,
            cache=cache,
        )
        if hypYaml:
            hypPath = Path(hypYaml).expanduser()
            if not hypPath.exists():
                raise FileNotFoundError(hypPath)
            with hypPath.open() as fp:
                self.kw.update(yaml.safe_load(fp))

    def filterOnePerClass(self, boxes):
        """
        Keep every detection for normal classes (stripe/solid) but
        ensure at most ONE cue (1) and ONE black (0) -- the one with
        the highest confidence score.
        Works with Ultralytics Boxes objects or plain dicts that have .cls and .conf
        """
        one_only = {0, 1}  # classes constrained to ≤1 instance
        best = {}  # cls → box with highest conf
        kept = []  # boxes for unconstrained classes

        for b in boxes:
            cls = int(b.cls)
            if cls in one_only:
                # keep the best-confidence box we’ve seen so far
                if cls not in best or float(b.conf) > float(best[cls].conf):
                    best[cls] = b
            else:
                kept.append(b)  # stripes/solids: keep them all

        # merge and re-sort by confidence (optional but nice)
        kept.extend(best.values())
        kept.sort(key=lambda x: float(x.conf), reverse=True)
        return kept

    def onePerClassCallback(self, trainer):
        print("$$$$$$$$$ CALLED onePerClassCallback $$$$$$$$$")
        for r in trainer.pred:
            r.boxes = self.filterOnePerClass(r.boxes)

    def train(self):
        print("[UltraTrainer] starting training with:", self.kw)
        model = YOLO(self.model)
        # model.add_callback("on_predict_postprocess_end", self.onePerClassCallback)
        model.train(**self.kw)


# ──────────────────────────────────────────────────────────
# 5.  Orchestration / CLI
# ──────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", help="override default JSON config")
    ap.add_argument(
        "--use-transformed",
        action="store_true",
        help="use transformed dataset (default: use original pix2pockets)",
    )
    args = ap.parse_args()

    MODEL_NAME = "pix2pockets_remapped_transformed"
    CONFIG = {
        "srcImgDir": f"data/{MODEL_NAME}/images",
        "srcLblDir": f"data/{MODEL_NAME}/labels",
        "dstRoot": "/tmp/workdir",
        "oldToNewMap": {3: 3, 2: 2, 1: 1, 0: 0}, # shotstudio
        #"oldToNewMap": {4: 3, 3: 2, 1: 1, 0: 0},  # pix2pocket → shotstudio, not needed 
        "classNames": ["black", "cue", "solid", "stripe"],  # id 0→black …
        "split": [0.8, 0.10, 0.10],
        "trainer": {
            "model": "yolov8n.pt",  # or yolov8s.pt, yolov9c.pt …
            "hyp": "data/hyps/hyp.custom.yaml",  # optional
            "epochs": 30,
            "imgsz": 1280,
            "batch": 4,  # Further reduced batch size for stability
            "device": "mps",
            "workers": 8,  # Reduced workers to match batch size
            "project": "tableizer",
            "name": MODEL_NAME,
            "cache": True,
            "patience": 10,  # Early stopping patience
            "save_period": 5,  # Save checkpoint every 5 epochs
        },
    }
    # ---------- default config ----------
    if args.use_transformed:
        # Use transformed dataset
        CONFIG["srcImgDir"] = "data/pix2pockets_transformed/images"
        CONFIG["srcLblDir"] = "data/pix2pockets_transformed/labels"
        # Labels are already remapped in transformed dataset
        CONFIG["oldToNewMap"] = ({0: 0, 1: 1, 2: 2, 3: 3},)

    # ---------- override ----------
    if args.config:
        CONFIG.update(json.loads(Path(args.config).read_text()))

    srcImgDir, srcLblDir = map(
        lambda p: Path(p).expanduser(), (CONFIG["srcImgDir"], CONFIG["srcLblDir"])
    )
    dstRoot = Path(CONFIG["dstRoot"]).expanduser()
    dstRoot.mkdir(exist_ok=True)

    # 1. remap labels
    LabelRemapper(srcImgDir, srcLblDir, dstRoot, CONFIG["oldToNewMap"]).run()
    # 2. split
    DataSplitter(
        dstRoot / "images_all",
        dstRoot / "labels_all",
        dstRoot,
        split=tuple(CONFIG["split"]),
    ).run()
    # 3. yaml
    dataYaml = YamlWriter(dstRoot, CONFIG["classNames"]).write()
    # 4. train
    t = CONFIG["trainer"]
    UltraTrainer(
        model=t["model"],
        dataYaml=dataYaml,
        hypYaml=t["hyp"],
        epochs=t["epochs"],
        imgsz=t["imgsz"],
        batch=t["batch"],
        device=t["device"],
        workers=t["workers"],
        project=t["project"],
        name=t["name"],
    ).train()


if __name__ == "__main__":
    main()
