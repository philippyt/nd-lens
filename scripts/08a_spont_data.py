import json
import tarfile
from pathlib import Path

import pandas as pd
from norsk_asr.text import normalize

from dialect_probing.config import load_config
from dialect_probing.spont import NBTALE_AREAS, SAMTALE_AREAS, module3_segments
from dialect_probing.written_standard import classify, load_lexicons

SAMTALE_DROP = {"%unint", "%foreign", "%unk", "%overlapping"}

def samtale(root):
    audio = root / "audio"
    for tar in sorted((root / "hf" / "data").glob("*_bm_*.tar.gz")):
        if not (audio / tar.name).with_suffix("").with_suffix(".done").exists():
            with tarfile.open(tar) as t:
                t.extractall(audio, filter="data")
            (audio / tar.name).with_suffix("").with_suffix(".done").touch()
    rows = []
    for meta in sorted((root / "hf" / "data").glob("*_metadata.jsonl")):
        for line in meta.read_text().splitlines():
            r = json.loads(line)
            if r["orthography"] != "bm":
                continue
            rows.append({"set": "nb_samtale", "seg_id": r["file_name"], "speaker": r["speaker_id"], "dialect": r["dialect"],
                         "area": SAMTALE_AREAS[r["dialect"]], "path": str(audio / r["file_name"]), "start": 0.0,
                         "end": r["duration"], "ref_raw": r["orthographic"], "source_type": r["source_type"],
                         "overlap": r["overlap_previous"] or r["overlap_next"]})
    df = pd.DataFrame(rows)
    bad = df["ref_raw"].str.contains("|".join(SAMTALE_DROP))
    print(f"NB Samtale Bokmål segments {len(df)}, dropped for {sorted(SAMTALE_DROP)}: {bad.sum()}")
    df = df[~bad]
    df["ref_raw"] = df["ref_raw"].str.replace(r"%\w+", " ", regex=True)
    return df

def main():
    cfg = load_config()
    nb = cfg["data_dir"] / "nbtale"
    out = cfg["data_dir"] / "spont"
    out.mkdir(exist_ok=True)
    (cfg["out_dir"] / "spont").mkdir(exist_ok=True)

    m3, stats = module3_segments(nb / "raw" / "Annotation.zip")
    print("NB Tale module 3 parse:", stats)
    spk = pd.read_parquet(nb / "speakers.parquet")
    m3 = m3.merge(spk[["speaker", "group", "native"]], on="speaker", how="left")
    print(f"recordings {m3['recording'].nunique()}, speakers {m3['speaker'].nunique()}, "
          f"with speaker info {m3['group'].notna().groupby(m3['speaker']).first().sum()}, native {m3.groupby('speaker')['native'].first().sum()}")
    m3 = m3[m3["native"] == True]
    m3 = m3.assign(set="nbtale_m3", area=m3["group"].map(NBTALE_AREAS).fillna("mixed"),
                   path=m3["recording"].map(lambda r: str(nb / "raw" / f"{r}.wav")))
    missing = ~m3["path"].map(lambda p: Path(p).exists())
    print(f"module 3 native segments {len(m3)}, missing audio {missing.sum()}")
    m3 = m3[~missing]

    nob, nno = load_lexicons(cfg["data_dir"] / "ordbank")
    ws = m3.groupby("speaker")["ref_raw"].apply(" ".join).map(lambda t: classify(t, nob, nno)[0])
    print("module 3 written standard per speaker:", ws.value_counts().to_dict())

    st = samtale(cfg["data_dir"] / "nb_samtale")
    seg = pd.concat([m3[["set", "seg_id", "speaker", "group", "area", "path", "start", "end", "ref_raw"]],
                     st.drop(columns=["dialect"])], ignore_index=True)
    seg["ref"] = seg["ref_raw"].map(normalize)
    seg = seg[seg["ref"].str.len() > 0]
    seg.to_parquet(out / "segments.parquet")
    seg["hours"] = (seg["end"] - seg["start"]) / 3600
    print(seg.groupby(["set", "area"]).agg(speakers=("speaker", "nunique"), segments=("seg_id", "size"),
                                           hours=("hours", "sum"), words=("ref", lambda s: s.str.split().str.len().sum())).round(2).to_string())
    print("NB Samtale overlap-flagged segments:", int(st["overlap"].sum()), "of", len(st))
    print("segments with digits in the raw reference:", seg.groupby("set")["ref_raw"].apply(lambda s: s.str.contains(r"\d").sum()).to_dict())

if __name__ == "__main__":
    main()
