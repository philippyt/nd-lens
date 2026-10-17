from collections import defaultdict

import jiwer
import numpy as np
import pandas as pd
from norsk_asr.text import normalize

from dialect_probing.config import load_config
from dialect_probing.written_standard import LEXICONS

B = 10000
MIN_REL_GAP = 0.20
METRICS = {"wer": ("err", "words"), "cer": ("cerr", "chars"), "misheard": ("misheard", "words")}

def lemma_map(ordbank_dir) -> dict[str, set]:
    lem = defaultdict(set)
    with open(ordbank_dir / LEXICONS["nob"], encoding="latin-1") as f:
        next(f)
        for line in f:
            c = line.split("\t")
            if len(c) > 2:
                lem[c[2].lower()].add(c[1])
    return lem

def counts(ref: str, hyp: str, lem: dict) -> dict:
    hyp = " ".join((hyp or "").split())
    h_text = hyp or "<empty>"
    o = jiwer.process_words(ref, h_text)
    mis = 0
    r, h = ref.split(), h_text.split()
    for c in o.alignments[0]:
        if c.type == "substitute":
            for rw, hw in zip(r[c.ref_start_idx:c.ref_end_idx], h[c.hyp_start_idx:c.hyp_end_idx]):
                if not (lem.get(rw, set()) & lem.get(hw, set())):
                    mis += 1
    ch = jiwer.process_characters(ref, hyp or " ")
    return {"words": o.hits + o.substitutions + o.deletions, "err": o.substitutions + o.deletions + o.insertions,
            "sub": o.substitutions, "del": o.deletions, "ins": o.insertions, "misheard": mis,
            "chars": ch.hits + ch.substitutions + ch.deletions, "cerr": ch.substitutions + ch.deletions + ch.insertions,
            "exact": int(ref == hyp)}

def boot(spk: pd.DataFrame, col: str, denom: str, rng) -> np.ndarray:
    a = spk[[col, denom]].to_numpy(float)
    idx = rng.integers(0, len(a), (B, len(a)))
    s = a[idx].sum(1)
    return s[:, 0] / s[:, 1]

def main():
    cfg = load_config()
    out = cfg["out_dir"] / "spont"
    seg = pd.read_parquet(cfg["data_dir"] / "spont" / "segments.parquet")
    hyp = pd.concat([pd.read_parquet(p) for p in sorted(out.glob("hyp_*.parquet"))])
    lem = lemma_map(cfg["data_dir"] / "ordbank")
    d = hyp.merge(seg[["seg_id", "set", "speaker", "area", "ref"]], on="seg_id")
    d["hyp_n"] = d["hyp"].fillna("").map(normalize)
    c = pd.DataFrame([counts(r, h, lem) for r, h in zip(d["ref"], d["hyp_n"])], index=d.index)
    d = pd.concat([d, c], axis=1)
    d.drop(columns=["ref"]).to_parquet(out / "scored_segments.parquet")
    cols = ["words", "err", "sub", "del", "ins", "misheard", "chars", "cerr", "exact"]
    spk = d.groupby(["set", "model", "area", "speaker"])[cols].sum().reset_index()
    spk["segments"] = d.groupby(["set", "model", "area", "speaker"]).size().to_numpy()

    rng = np.random.default_rng(0)
    rows, gaps = [], []
    for (st, m), g in spk.groupby(["set", "model"]):
        tot = g[cols].sum()
        rows.append({"set": st, "model": m, "area": "all", "speakers": len(g), "words": tot["words"],
                     "wer": tot["err"] / tot["words"], "cer": tot["cerr"] / tot["chars"], "misheard": tot["misheard"] / tot["words"],
                     "exact_match": tot["exact"] / g["segments"].sum()})
        bs = {}
        for area, ga in g.groupby("area"):
            if area == "mixed":
                continue
            r = {"set": st, "model": m, "area": area, "speakers": len(ga), "words": ga["words"].sum()}
            for name, (col, den) in METRICS.items():
                b = boot(ga, col, den, rng)
                bs[(area, name)] = b
                r[name] = ga[col].sum() / ga[den].sum()
                r[f"{name}_lo"], r[f"{name}_hi"] = np.percentile(b, [2.5, 97.5])
            r["exact_match"] = ga["exact"].sum() / ga["segments"].sum()
            rows.append(r)
        gr = {"set": st, "model": m}
        for name, (col, den) in METRICS.items():
            w, e = bs[("west", name)], bs[("east", name)]
            ow, oe = (g.loc[g["area"] == a, col].sum() / g.loc[g["area"] == a, den].sum() for a in ("west", "east"))
            gr[f"{name}_gap"], gr[f"{name}_rel"] = ow - oe, (ow - oe) / oe
            gr[f"{name}_gap_lo"], gr[f"{name}_gap_hi"] = np.percentile(w - e, [2.5, 97.5])
            gr[f"{name}_rel_lo"], gr[f"{name}_rel_hi"] = np.percentile((w - e) / e, [2.5, 97.5])
        gr["meaningful"] = bool(gr["wer_rel"] >= MIN_REL_GAP and gr["wer_gap_lo"] > 0 and gr["misheard_gap_lo"] > 0)
        gaps.append(gr)
    areas = pd.DataFrame(rows)
    gaps = pd.DataFrame(gaps)
    areas.to_csv(out / "areas.csv", index=False)
    gaps.to_csv(out / "gaps.csv", index=False)

    pct = lambda x: f"{100 * x:5.1f}"
    for st, g in areas.groupby("set"):
        print(f"\n{st}: WER % [95% CI] by area")
        for m, gm in g.groupby("model"):
            parts = []
            for area in ("east", "west", "north", "central", "south", "all"):
                r = gm[gm["area"] == area]
                if len(r):
                    r = r.iloc[0]
                    ci = f" [{pct(r['wer_lo']).strip()},{pct(r['wer_hi']).strip()}]" if area != "all" else ""
                    parts.append(f"{area} {pct(r['wer']).strip()}{ci}")
            print(f"  {m:18s} " + "  ".join(parts))
    print("\nWest minus East (points, relative) with 95% CI:")
    for _, r in gaps.iterrows():
        print(f"  {r['set']:10s} {r['model']:18s} WER {100 * r['wer_gap']:+.1f} [{100 * r['wer_gap_lo']:+.1f},{100 * r['wer_gap_hi']:+.1f}] "
              f"({r['wer_rel']:+.0%} [{r['wer_rel_lo']:+.0%},{r['wer_rel_hi']:+.0%}])  CER {100 * r['cer_gap']:+.1f}  "
              f"misheard {100 * r['misheard_gap']:+.1f} [{100 * r['misheard_gap_lo']:+.1f},{100 * r['misheard_gap_hi']:+.1f}]  "
              f"meaningful {r['meaningful']}")
    print("\nexact-match segment rate (contamination check):")
    print(areas[areas["area"] == "all"].pivot(index="model", columns="set", values="exact_match").round(3).to_string())

    st = d[d["set"] == "nb_samtale"].assign(split=lambda x: x["seg_id"].str.split("/").str[1])
    sp = st.groupby(["model", "split"])[["err", "words", "exact"]].sum()
    sp["wer"] = sp["err"] / sp["words"]
    sp["exact_rate"] = sp["exact"] / st.groupby(["model", "split"]).size()
    rows = []
    for m, g in st.groupby("model"):
        s = g.groupby(["speaker", "split"])[["err", "words"]].sum().unstack("split").dropna()
        tr, te = s[("err", "train")] / s[("words", "train")], s[("err", "test")] / s[("words", "test")]
        rows.append({"model": m, "speakers_in_both": len(s), "median_test_minus_train_wer": float((te - tr).median()),
                     "share_speakers_test_worse": float((te > tr).mean())})
    split_check = pd.DataFrame(rows)
    sp.reset_index().to_csv(out / "samtale_split_check.csv", index=False)
    print("\nNB Samtale train vs test split (splits share speakers; training on train would show as lower train WER):")
    print(sp[["wer", "exact_rate"]].unstack("split").round(3).to_string())
    print(split_check.round(3).to_string(index=False))

    m3 = spk[spk["set"] == "nbtale_m3"].merge(seg[["speaker", "group"]].drop_duplicates(), on="speaker")
    grp = m3.groupby(["model", "group"]).agg(speakers=("speaker", "nunique"), words=("words", "sum"), err=("err", "sum"))
    grp["wer"] = grp["err"] / grp["words"]
    grp.reset_index().to_csv(out / "module3_groups.csv", index=False)
    print("\nNB Tale module 3 WER % per group (speakers):")
    t = grp["wer"].unstack("group") * 100
    print(t.round(1).to_string())
    print("speakers per group:", grp.xs(grp.index.levels[0][0])["speakers"].to_dict())

if __name__ == "__main__":
    main()
