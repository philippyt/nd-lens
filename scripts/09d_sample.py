import json

import jiwer
import numpy as np
import pandas as pd

from dialect_probing.config import load_config

PER_AREA = 90
PER_SPEAKER = 8
MIN_S, MAX_S = 30, 240
PILOT = 20

def main():
    cfg = load_config()
    root = cfg["data_dir"] / "npsc"
    out = cfg["out_dir"] / "npsc"
    sent = pd.read_parquet(root / "sentences.parquet").merge(pd.read_parquet(out / "hyp_norsk-asr.parquet")[["sentence_id", "lm"]], on="sentence_id")
    sent = sent.sort_values(["meeting_date", "sentence_order"])
    text = sent.groupby("speech_id").agg(reference=("ref", " ".join), asr=("lm", " ".join), nn=("is_nn", "mean"))
    sp = pd.read_parquet(root / "speeches.parquet").drop(columns=["ref_text"]).merge(text, on="speech_id")
    sp = sp[(sp["kind"] != "procedural") & sp["area"].notna() & sp["seconds"].between(MIN_S, MAX_S)]
    rng = np.random.default_rng(0)
    picks = []
    for area, g in sp.groupby("area"):
        g = g.sample(frac=1, random_state=int(rng.integers(1 << 31)))
        g = g.groupby("speaker_name", group_keys=False).head(PER_SPEAKER)
        picks.append(g.head(PER_AREA))
    s = pd.concat(picks).sample(frac=1, random_state=0).reset_index(drop=True)
    s["wer"] = [jiwer.wer(r, h) for r, h in zip(s["reference"], s["asr"])]
    s["pilot"] = False
    s.loc[s.groupby("area").head(PILOT // 5).index, "pilot"] = True
    cols = ["speech_id", "meeting_date", "data_split", "speaker_name", "area", "party", "kind", "seconds", "nn", "wer", "reference", "asr", "pilot"]
    s[cols].to_parquet(out / "sample.parquet")
    with open(out / "sample.jsonl", "w", encoding="utf-8") as f:
        for r in s[cols].sort_values("pilot", ascending=False).to_dict("records"):
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    print(f"sample {len(s)} speeches, pilot {s['pilot'].sum()}")
    print(s.groupby("area").agg(speeches=("speech_id", "size"), speakers=("speaker_name", "nunique"),
                                minutes=("seconds", lambda x: x.sum() / 60), replikk=("kind", lambda x: (x == "replikk").sum()),
                                wer=("wer", "median")).round(2).to_string())

if __name__ == "__main__":
    main()
