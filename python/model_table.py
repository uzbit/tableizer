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
# yolo export model=/Users/uzbit/Documents/projects/tableizer/tableizer/exp2/weights/best.pt format=torchscript device=cpu imgsz=800 nms=False agnostic_nms=False

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
        self.srcImgDir, self.srcLblDir = srcImgDir, srcLblDir
        self.dstRoot, self.oldToNewMap = dstRoot, oldToNewMap
        self.imgExts = imgExts
        self.dstImgDir = dstRoot / "images_all"
        self.dstLblDir = dstRoot / "labels_all"
        self.dstImgDir.mkdir(parents=True, exist_ok=True)
        self.dstLblDir.mkdir(parents=True, exist_ok=True)

    def _remap_file(self, lblPath: Path, imgPath: Path) -> bool:
        outLines: List[str] = []
        for line in lblPath.read_text().splitlines():
            if not line.strip():
                continue
            oldId, *rest = line.split()
            oldId = int(oldId)
            if oldId not in self.oldToNewMap:  # e.g. diamonds
                continue
            newId = self.oldToNewMap[oldId]
            outLines.append(" ".join([str(newId), *rest]))
        if not outLines:
            return False  # skip images with no target objects

        shutil.copy2(imgPath, self.dstImgDir / imgPath.name)
        (self.dstLblDir / lblPath.name).write_text("\n".join(outLines) + "\n")
        return True

    def run(self) -> None:
        kept = dropped = 0
        for lbl in sorted(self.srcLblDir.glob("*.txt")):
            # find corresponding image
            for ext in self.imgExts:
                imgPath = self.srcImgDir / f"{lbl.stem}{ext}"
                if imgPath.exists():
                    break
            else:
                print(f"[WARN] missing image for {lbl}")
                continue

            # remap_file returns True if the label survives
            if self._remap_file(lbl, imgPath):
                kept += 1          # count this image as kept
            else:
                dropped += 1       # label had no target classes

        print(f"[LabelRemapper] kept {kept} imgs, dropped {dropped}")

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
            shutil.move(str(self.allImgDir / f"{s}.jpg"),
                        self.dstRoot / f"images/{subset}/{s}.jpg")
            shutil.move(str(self.allLblDir / f"{s}.txt"),
                        self.dstRoot / f"labels/{subset}/{s}.txt")

    def run(self):
        train, val, test = [], [], []
        for stems in self._collect().values():
            random.shuffle(stems)
            n = len(stems)
            nTr, nVal = int(self.split[0]*n), int(self.split[1]*n)
            train += stems[:nTr]
            val   += stems[nTr:nTr+nVal]
            test  += stems[nTr+nVal:]
        self._dump("train", train)
        self._dump("val",   val)
        self._dump("test",  test)
        print(f"[DataSplitter] train:{len(train)}  val:{len(val)}  test:{len(test)}")

# ──────────────────────────────────────────────────────────
# 3.  data.yaml writer
# ──────────────────────────────────────────────────────────
class YamlWriter:
    def __init__(self, root: Path, names: List[str]): self.root, self.names = root, names
    def write(self) -> Path:
        path = self.root / "data.yaml"
        yaml = {
            "path": str(self.root),
            "train": "images/train",
            "val":   "images/val",
            "test":  "images/test",
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
        imgsz=800,
        batch=16,
        device="mps",          # "cpu", "mps", "0", "0,1"
        workers=40,
        project="tableizer",
        name="exp",
        cache=True,
    ):
        self.model, self.dataYaml, self.hypYaml = model, dataYaml, hypYaml
        self.kw = dict(
            data   = str(dataYaml),
            epochs = epochs,
            imgsz  = imgsz,
            batch  = batch,
            device = device,
            workers= workers,
            project= project,
            name   = name,
            cache  = cache,
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
        one_only = {0, 1}                # classes constrained to ≤1 instance
        best     = {}                    # cls → box with highest conf
        kept     = []                    # boxes for unconstrained classes

        for b in boxes:
            cls = int(b.cls)
            if cls in one_only:
                # keep the best-confidence box we’ve seen so far
                if cls not in best or float(b.conf) > float(best[cls].conf):
                    best[cls] = b
            else:
                kept.append(b)           # stripes/solids: keep them all

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
    args = ap.parse_args()

    # ---------- default config ----------
    CONFIG = {
        "srcImgDir": "data/pix2pockets/images",
        "srcLblDir": "data/pix2pockets/labels",
        "dstRoot":   "/tmp/workdir",
        # original-id → new-id
        "oldToNewMap": {4:3, 3:2, 1:1, 0:0},     # adjust as needed
        "classNames": ["black", "cue", "solid", "stripe"],  # id 0→black …
        "split": [0.8, 0.15, 0.05],
        "trainer": {
            "model":   "yolov9s.pt",        # or yolov8s.pt, yolov9c.pt …
            "hyp":     "data/hyps/hyp.custom.yaml",  # optional
            "epochs":  100,
            "imgsz":   800,
            "batch":   20,
            "device":  "mps",
            "workers": 16,
            "project": "tableizer",
            "name":    "exp",
        },
    }
    # ---------- override ----------
    if args.config:
        CONFIG.update(json.loads(Path(args.config).read_text()))

    srcImgDir, srcLblDir = map(lambda p: Path(p).expanduser(),
                               (CONFIG["srcImgDir"], CONFIG["srcLblDir"]))
    dstRoot = Path(CONFIG["dstRoot"]).expanduser(); dstRoot.mkdir(exist_ok=True)

    # 1. remap labels
    LabelRemapper(srcImgDir, srcLblDir, dstRoot, CONFIG["oldToNewMap"]).run()
    # 2. split
    DataSplitter(dstRoot/"images_all", dstRoot/"labels_all", dstRoot,
                 split=tuple(CONFIG["split"])).run()
    # 3. yaml
    dataYaml = YamlWriter(dstRoot, CONFIG["classNames"]).write()
    # 4. train
    t = CONFIG["trainer"]
    UltraTrainer(
        model   = t["model"],
        dataYaml= dataYaml,
        hypYaml = t["hyp"],
        epochs  = t["epochs"],
        imgsz   = t["imgsz"],
        batch   = t["batch"],
        device  = t["device"],
        workers = t["workers"],
        project = t["project"],
        name    = t["name"],
    ).train()

if __name__ == "__main__":
    main()