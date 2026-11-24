"""Public-data acquisition, annotation cleaning and label-blind duplicate grouping."""

from pathlib import Path
import json, re, hashlib, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import requests
import ftfy
from PIL import Image, ImageOps
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
REVISION = "10a5626ba112ab9c50369c473a2062bcf205d708"
BASE = f"https://huggingface.co/datasets/QCRI/CrisisMMD/resolve/{REVISION}/"


def clean_text(text):
    text = ftfy.fix_text(str(text))
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+|^RT\s*:?\s*", " ", text)
    return re.sub(r"\s+", " ", text.replace("#", " ").lower()).strip()


def download():
    raw = ROOT / "data/raw"
    (raw / "informative").mkdir(parents=True, exist_ok=True)
    rows = []
    for split in ["train", "dev", "test"]:
        target = raw / f"informative/{split}.json"
        if not target.exists():
            response = requests.get(BASE + f"informative/{split}.json", timeout=120)
            response.raise_for_status()
            target.write_bytes(response.content)
        obj = json.loads(target.read_text())
        rows.extend(obj if isinstance(obj, list) else obj["data"])
    annotations = raw / "annotations"
    annotations.mkdir(exist_ok=True)
    for event in sorted({row["event_name"] for row in rows}):
        target = annotations / f"{event}.tsv"
        if not target.exists():
            response = requests.get(
                BASE + f"files_individual_events/{event}_final_data.tsv", timeout=120
            )
            response.raise_for_status()
            target.write_bytes(response.content)

    def fetch(row):
        path = raw / row["image_path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            return
        for attempt in range(6):
            try:
                r = requests.get(BASE + row["image_path"], timeout=90)
                r.raise_for_status()
                tmp = path.with_suffix(".partial")
                tmp.write_bytes(r.content)
                tmp.replace(path)
                return
            except Exception:
                if attempt == 5:
                    raise
                time.sleep(2**attempt)

    with ThreadPoolExecutor(max_workers=12) as pool:
        for i, _ in enumerate(pool.map(fetch, rows), 1):
            if i % 500 == 0:
                print(f"Downloaded or verified {i}/{len(rows)}", flush=True)


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def prepare():
    rows = []
    for split in ["train", "dev", "test"]:
        obj = json.loads((ROOT / f"data/raw/informative/{split}.json").read_text())
        records = obj if isinstance(obj, list) else obj["data"]
        rows.extend(dict(row, original_split=split) for row in records)
    frame = pd.DataFrame(rows)
    frame["text"] = frame.tweet_text.map(clean_text)
    original = pd.concat(
        [
            pd.read_csv(p, sep="\t")
            for p in sorted((ROOT / "data/raw/annotations").glob("*.tsv"))
        ],
        ignore_index=True,
    )
    original = original.set_index("image_path")
    assert original.index.is_unique
    frame["mirror_label"] = frame.label
    frame["label"] = frame.image_path.map(original.image_info)
    frame["text_label"] = frame.image_path.map(original.text_info)
    assert frame.label.notna().all() and set(frame.label) == {
        "informative",
        "not_informative",
    }
    frame["y"] = (frame.label == "informative").astype(int)

    def inspect(path):
        with Image.open(ROOT / "data/raw" / path) as source:
            im = ImageOps.exif_transpose(source).convert("RGB")
            arr = np.asarray(im)
            digest = hashlib.sha256(str(im.size).encode() + arr.tobytes()).hexdigest()
            return digest, im.width, im.height, *arr.mean(axis=(0, 1)).tolist()

    with ThreadPoolExecutor(max_workers=8) as pool:
        stats = list(pool.map(inspect, frame.image_path))
    frame[["pixel_hash", "width", "height", "red", "green", "blue"]] = pd.DataFrame(
        stats, index=frame.index
    )
    uf = UnionFind(len(frame))
    for column in ["tweet_id", "pixel_hash", "text"]:
        seen = {}
        for i, value in enumerate(frame[column]):
            if column == "text" and len(value) < 20:
                continue
            if value in seen:
                uf.union(i, seen[value])
            else:
                seen[value] = i
    frame["group"] = [uf.find(i) for i in range(len(frame))]
    frame["split"] = "train"
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (_, heldout) in enumerate(cv.split(frame, frame.y, frame.group)):
        if fold < 2:
            frame.loc[heldout, "split"] = "test" if fold == 0 else "val"
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        assert not set(frame.loc[frame.split == a, "group"]) & set(
            frame.loc[frame.split == b, "group"]
        )
    frame.to_csv(ROOT / "data/manifest.csv", index=False)
    audit = {
        "rows": len(frame),
        "mirror_label_disagreements": int((frame.mirror_label != frame.label).sum()),
        "image_text_label_disagreements": int((frame.text_label != frame.label).sum()),
        "groups": int(frame.group.nunique()),
        "tweet_ids": int(frame.tweet_id.nunique()),
        "pixel_duplicates": int(len(frame) - frame.pixel_hash.nunique()),
        "splits": pd.crosstab(frame.split, frame.label).to_dict(),
        "original_split_group_overlap": {},
    }
    for a, b in [("train", "dev"), ("train", "test"), ("dev", "test")]:
        audit["original_split_group_overlap"][f"{a}/{b}"] = len(
            set(frame.loc[frame.original_split == a, "group"])
            & set(frame.loc[frame.original_split == b, "group"])
        )
    audit["annotation_sha256"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((ROOT / "data/raw/annotations").glob("*.tsv"))
    }
    (ROOT / "results/data_audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))
    return frame
