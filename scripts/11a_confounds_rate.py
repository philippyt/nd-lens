import re
import xml.etree.ElementTree as ET
import zipfile
from collections import defaultdict

import jiwer
import numpy as np
import pandas as pd
from norsk_asr.text import normalize
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

from dialect_probing.config import load_config

PAUSES = {"sil", "inhale", "exhale", "fp", "vowel", "nasal"}
VOWELS = re.compile(r"[aeiouyæøåéèêóòôüäöáà]+")
AREAS = ["east", "west", "central", "north", "south"]
B = 10000

def pause_events(annotation_zip) -> dict:
    root = ET.fromstring(zipfile.ZipFile(annotation_zip).read("Annotation/part_3.xml"))
    ev = defaultdict(list)
    for ann in root.iter("annotation"):
        for el in ann.iter():
            if el.tag in PAUSES and el.get("start") and el.get("end"):
                ev[ann.get("id")].append((float(el.get("start")), float(el.get("end"))))
    return ev

def overlap(events, a: float, b: float) -> float:
    return sum(max(0.0, min(b, e) - max(a, s)) for s, e in events)

def boot_gap(spk: pd.DataFrame, rng) -> tuple[float, float, float]:
    w, e = spk[spk["area"] == "west"], spk[spk["area"] == "east"]
    if len(w) < 2 or len(e) < 2:
        return np.nan, np.nan, np.nan
    bw = w[["err", "words"]].to_numpy(float)[rng.integers(0, len(w), (B, len(w)))].sum(1)
    be = e[["err", "words"]].to_numpy(float)[rng.integers(0, len(e), (B, len(e)))].sum(1)
    d = bw[:, 0] / bw[:, 1] - be[:, 0] / be[:, 1]
    return w["err"].sum() / w["words"].sum() - e["err"].sum() / e["words"].sum(), *np.percentile(d, [2.5, 97.5])

def mixed(rows: pd.DataFrame, formula: str) -> pd.Series:
    m = BinomialBayesMixedGLM.from_formula(formula, {"speaker": "0 + C(speaker)"}, rows)
    r = m.fit_vb()
    return pd.Series(r.fe_mean, index=m.exog_names), pd.Series(r.fe_sd, index=m.exog_names)

def main():
    cfg = load_config()
    out = cfg["out_dir"] / "spont"
    seg = pd.read_parquet(cfg["data_dir"] / "spont" / "segments.parquet")
    seg = seg[(seg["set"] == "nbtale_m3") & seg["area"].isin(AREAS)].copy()
    seg["recording"] = seg["seg_id"].str.split("@").str[0]
    spk = pd.read_parquet(cfg["data_dir"] / "nbtale" / "speakers.parquet")[["speaker", "gender", "birthyear", "age_group", "recording_place"]]
    seg = seg.merge(spk, on="speaker", how="left")
    ev = pause_events(cfg["data_dir"] / "nbtale" / "raw" / "Annotation.zip")
    seg["duration"] = seg["end"] - seg["start"]
    seg["pause"] = [overlap(ev[r], s, e) for r, s, e in zip(seg["recording"], seg["start"], seg["end"])]
    seg["syllables"] = seg["ref"].map(lambda t: sum(len(VOWELS.findall(w)) or 1 for w in t.split()))
    seg["speaking_rate"] = seg["syllables"] / seg["duration"]
    seg["articulation_rate"] = seg["syllables"] / (seg["duration"] - seg["pause"]).clip(lower=1)
    hyp = pd.read_parquet(out / "hyp_norsk-asr.parquet")
    seg = seg.merge(hyp[hyp["model"] == "norsk-asr greedy"][["seg_id", "hyp"]], on="seg_id")
    cnt = [jiwer.process_words(r, normalize(h or "") or "<empty>") for r, h in zip(seg["ref"], seg["hyp"])]
    seg["words"] = [o.hits + o.substitutions + o.deletions for o in cnt]
    seg["err"] = [o.substitutions + o.deletions + o.insertions for o in cnt]
    seg["sd"] = [o.substitutions + o.deletions for o in cnt]
    seg.drop(columns=["hyp"]).to_parquet(out / "m3_rate_segments.parquet")

    sp = seg.groupby(["speaker", "area", "gender", "recording_place"], dropna=False).agg(
        words=("words", "sum"), err=("err", "sum"), syllables=("syllables", "sum"), duration=("duration", "sum"),
        pause=("pause", "sum"), birthyear=("birthyear", "first"), age_group=("age_group", "first")).reset_index()
    sp["wer"] = sp["err"] / sp["words"]
    sp["speaking_rate"] = sp["syllables"] / sp["duration"]
    sp["articulation_rate"] = sp["syllables"] / (sp["duration"] - sp["pause"])
    sp["pause_share"] = sp["pause"] / sp["duration"]
    sp["age"] = 2014 - pd.to_numeric(sp["birthyear"], errors="coerce")
    sp.to_csv(out / "m3_speakers_rate.csv", index=False)

    print("1. confounds per area (module 3 natives)")
    t = sp.groupby("area").agg(speakers=("speaker", "size"), female=("gender", lambda g: (g == "F").mean()),
                               median_age=("age", "median"), older_group=("age_group", lambda g: (g.astype(str) == "2").mean()))
    print(t.round(2).to_string())
    print("\nrecording place by area (speakers):")
    print(pd.crosstab(sp["area"], sp["recording_place"]).to_string())
    print("\nWER % by recording place within area (speakers):")
    w = sp.groupby(["area", "recording_place"]).apply(lambda g: pd.Series({"speakers": len(g), "wer": 100 * g["err"].sum() / g["words"].sum()}))
    print(w.round(1).to_string())
    rng = np.random.default_rng(0)
    for name, sub in [("all speakers", sp), ("NTNU only", sp[sp["recording_place"] == "NTNU"]),
                      ("female", sp[sp["gender"] == "F"]), ("male", sp[sp["gender"] == "M"]),
                      ("age group 1 (16-39)", sp[sp["age_group"].astype(str) == "1"]), ("age group 2 (40+)", sp[sp["age_group"].astype(str) == "2"])]:
        v, lo, hi = boot_gap(sub, rng)
        print(f"West minus East, {name:20s}: {100 * v:+.1f} points [{100 * lo:+.1f}, {100 * hi:+.1f}]  "
              f"(West {(sub['area'] == 'west').sum()}, East {(sub['area'] == 'east').sum()} speakers)")

    print("\n2. speech rate per area (speaker means, syllables per second):")
    print(sp.groupby("area")[["speaking_rate", "articulation_rate", "pause_share", "wer"]].mean().round(3).to_string())
    for col in ("speaking_rate", "articulation_rate", "pause_share"):
        w_, e_ = sp.loc[sp["area"] == "west", col], sp.loc[sp["area"] == "east", col]
        print(f"  {col}: West - East {w_.mean() - e_.mean():+.3f}  (Spearman with speaker WER {sp[[col, 'wer']].corr('spearman').iloc[0, 1]:+.2f})")

    rows = []
    for r in seg.itertuples():
        rows.extend({"speaker": r.speaker, "area": r.area, "site": r.recording_place, "art": r.articulation_rate,
                     "y": int(i < r.sd)} for i in range(r.words))
    words = pd.DataFrame(rows)
    words["art_z"] = (words["art"] - words["art"].mean()) / words["art"].std()
    base = "y ~ C(area, Treatment('east'))"
    print(f"\nmixed logistic on {len(words)} reference words (error = substitution or deletion), speaker random effect:")
    res = {}
    for name, f in [("area", base), ("area + rate", base + " + art_z"), ("area + rate + site", base + " + art_z + C(site)")]:
        mean, sd = mixed(words, f)
        res[name] = mean
        keep = [k for k in mean.index if "west" in k or k == "art_z" or "site" in k]
        print(f"  {name:20s} " + "  ".join(f"{k.replace(chr(39), '').replace('C(area, Treatment(east))[T.', '').replace(']', '')} "
                                            f"{mean[k]:+.3f} [{mean[k] - 1.96 * sd[k]:+.3f},{mean[k] + 1.96 * sd[k]:+.3f}]" for k in keep))
    k = "C(area, Treatment('east'))[T.west]"
    print(f"  West log-odds reduced by rate: {1 - res['area + rate'][k] / res['area'][k]:.0%}; "
          f"by rate + site: {1 - res['area + rate + site'][k] / res['area'][k]:.0%}")
    lp = np.polyfit(sp["articulation_rate"], sp["wer"], 1, w=sp["words"])
    dr = sp.loc[sp["area"] == "west", "articulation_rate"].mean() - sp.loc[sp["area"] == "east", "articulation_rate"].mean()
    gap = sp.loc[sp["area"] == "west", "wer"].mean() - sp.loc[sp["area"] == "east", "wer"].mean()
    print(f"  linear check: WER slope {100 * lp[0]:+.2f} points per syllable/s x West-East rate difference {dr:+.2f} = "
          f"{100 * lp[0] * dr:+.2f} points of a {100 * gap:+.1f} point gap (speaker means)")

if __name__ == "__main__":
    main()
