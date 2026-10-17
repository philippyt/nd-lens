import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import pandas as pd

MAX_SECONDS = 30.0
DROP = {"anonymized", "incomprehensible"}
NBTALE_AREAS = {"g06": "west", "g07": "west", "g08": "west", "g09": "west", "g11": "east", "g12": "east",
                "g01": "north", "g02": "north", "g03": "north", "g04": "central", "g05": "central", "g10": "south"}
SAMTALE_AREAS = {"w": "west", "sw": "west", "e": "east", "n": "north", "t": "central"}

def _tokens(el: ET.Element, out: list, flags: dict):
    tag = el.tag
    if tag in DROP:
        flags["drop"] = True
    elif tag in ("word", "complex_word"):
        if el.get("deadend") == "1":
            flags["deadend"] += 1
        else:
            out.append(el.get("text"))
    elif tag == "slang":
        out.append(el.get("normalized"))
        flags["slang"] += 1
    elif tag == "multiword":
        for c in el:
            _tokens(c, out, flags)

def _sentences(ann: ET.Element) -> list[dict]:
    sents, words, flags, start = [], [], {"drop": False, "deadend": 0, "slang": 0}, 0.0
    for el in ann:
        if el.tag == "sentence_boundary" and el.get("end") is not None:
            t = float(el.get("end"))
            sents.append({"start": start, "end": t, "words": words, **flags})
            words, flags, start = [], {"drop": False, "deadend": 0, "slang": 0}, t
        else:
            _tokens(el, words, flags)
    if words:
        sents.append({"start": start, "end": float(ann.get("end")), "words": words, **flags})
    return sents

def module3_segments(annotation_zip: Path) -> tuple[pd.DataFrame, dict]:
    root = ET.fromstring(zipfile.ZipFile(annotation_zip).read("Annotation/part_3.xml"))
    rows, stats = [], {"sentences": 0, "dropped_marker": 0, "dropped_long": 0, "deadend_words": 0, "slang_words": 0}
    for ann in root.iter("annotation"):
        rec, spk = ann.get("id"), ann.get("speaker")
        cur = None
        for s in _sentences(ann):
            stats["sentences"] += 1
            stats["deadend_words"] += s["deadend"]
            stats["slang_words"] += s["slang"]
            bad = s["drop"] or s["end"] - s["start"] > MAX_SECONDS
            if s["drop"]:
                stats["dropped_marker"] += 1
            elif bad:
                stats["dropped_long"] += 1
            if bad or not s["words"]:
                if cur:
                    rows.append(cur)
                cur = None
                continue
            if cur and s["end"] - cur["start"] <= MAX_SECONDS:
                cur["end"], cur["words"] = s["end"], cur["words"] + s["words"]
            else:
                if cur:
                    rows.append(cur)
                cur = {"recording": rec, "speaker": spk, "start": s["start"], "end": s["end"], "words": s["words"]}
        if cur:
            rows.append(cur)
    df = pd.DataFrame(rows)
    df["ref_raw"] = df.pop("words").map(" ".join)
    df["seg_id"] = df["recording"] + "@" + df["start"].map(lambda x: f"{x:.2f}")
    return df, stats
