import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

URL = "https://www.nb.no/sbfil/talegjenkjenning/lingit"

REGIONS = {
    "g01": ("Finnmark", "nord"),
    "g02": ("Troms", "nord"),
    "g03": ("Nordland (Bodø, Rana)", "nord"),
    "g04": ("Helgeland, Namdal, Nordmøre", "trøndersk"),
    "g05": ("Trøndelag", "trøndersk"),
    "g06": ("Romsdal, Sunnmøre", "vest"),
    "g07": ("Sogn og Fjordane", "vest"),
    "g08": ("Rogaland + Hordaland", "vest"),
    "g09": ("Rogaland + Bergen", "vest"),
    "g10": ("Agder", "sør"),
    "g11": ("Østlandet utenom Oslo", "øst"),
    "g12": ("Oslo og Akershus", "øst"),
    "g24": ("Norsk, blandet (modul 2)", "blandet"),
}

SPEAKER_COLS = [
    "speaker", "part", "group", "gender", "age_group", "subid", "set_name", "recording_place", "birthyear",
    "height", "municipality", "native_language", "age_moved_to", "age_moved_from",
    "num_parents_with_same_dialect", "occupation", "education", "ethnicity",
]

def annotation_dir(raw: Path) -> Path:
    return Path(raw) / "docs" / "Annotation" / "Annotation"

def read_speakers(raw: Path) -> pd.DataFrame:
    frames = []
    for part in (1, 2):
        f = annotation_dir(raw) / f"speakerinfo_part_{part}.txt"
        rows = [line.rstrip("\n").split("\t") for line in f.open(encoding="utf-8") if line.strip() and not line.startswith("#")]
        frames.append(pd.DataFrame([r[: len(SPEAKER_COLS)] for r in rows], columns=SPEAKER_COLS))
    df = pd.concat(frames, ignore_index=True)
    df["native"] = df["native_language"].eq("_")
    df["region"] = df["group"].map(lambda g: REGIONS.get(g, ("", ""))[0])
    df["macro_region"] = df["group"].map(lambda g: REGIONS.get(g, ("", ""))[1])
    return df

def read_annotations(raw: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    utts, words = [], []
    for part in (1, 2):
        root = ET.parse(annotation_dir(raw) / f"part_{part}.xml").getroot()
        for ann in root.iter("annotation"):
            uid = ann.get("id")
            utts.append({
                "utt_id": uid,
                "speaker": ann.get("speaker"),
                "group": uid.split("/")[1].replace("group_", "g"),
                "sentence_set": uid.rsplit("-", 1)[1][0],
                "sentence_id": uid.rsplit("-", 1)[1],
                "text": ann.get("text"),
                "ann_end": float(ann.get("end")),
            })
            for i, w in enumerate(ann.iter("word")):
                text = w.get("text")
                segs = w.findall("seg")
                if not segs or text.startswith("<"):
                    continue
                words.append({
                    "utt_id": uid, "word_idx": i, "word_orig": text,
                    "ref_start": float(segs[0].get("start")), "ref_end": float(segs[-1].get("end")),
                })
    return pd.DataFrame(utts), pd.DataFrame(words)

def wav_path(raw: Path, utt_id: str, mic: str = "sennheiser") -> Path:
    return Path(raw) / "Speech" / mic / f"{utt_id}.wav"
