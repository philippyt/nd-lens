import re

import jiwer
import numpy as np
import pandas as pd

from dialect_probing.config import load_config

NEG = {"ikke", "ikkje", "ikkj", "itte", "ente", "inte", "itj", "ittj", "ikk", "kje", "aldri", "ingen", "ingenting",
       "inga", "intet", "verken", "heller"}
UNITS = "en|ett|to|tre|fire|fem|seks|sju|syv|åtte|ni"
NUM = re.compile(rf"^(null|to|tre|fire|fem|seks|sju|syv|åtte|ni|ti|elleve|tolv|tretten|fjorten|femten|seksten|sytten|"
                 rf"atten|nitten|((tjue|tretti|førti|femti|seksti|sytti|åtti|nitti)({UNITS})?)|hundre|tusen|millioner?|"
                 rf"milliarder?|prosent|komma)$")
AREAS = ["east", "south", "north", "mid", "west"]
B = 10000

def kind(w: str) -> str | None:
    if w in NEG and w != "heller":
        return "negation"
    if NUM.match(w):
        return "number"
    return None

def outcomes(ref: str, hyp: str) -> list[tuple[str, str]]:
    r, h = ref.split(), (hyp.split() or ["<empty>"])
    o = jiwer.process_words(ref, " ".join(h))
    res = []
    for c in o.alignments[0]:
        rw = r[c.ref_start_idx:c.ref_end_idx]
        hw = h[c.hyp_start_idx:c.hyp_end_idx]
        for i, w in enumerate(rw):
            k = kind(w)
            if not k:
                continue
            if c.type == "equal":
                res.append((k, "preserved"))
            elif c.type == "delete":
                res.append((k, "dropped"))
            else:
                alt = hw[i] if i < len(hw) else ""
                same = kind(alt) == k
                res.append((k, "variant" if same and k == "negation" else ("substituted_same_kind" if same else "substituted")))
    return res

def main():
    cfg = load_config()
    root = cfg["data_dir"] / "npsc"
    out = cfg["out_dir"] / "npsc"
    sent = pd.read_parquet(root / "sentences.parquet")
    hyp = pd.read_parquet(out / "hyp_norsk-asr.parquet")
    spk = pd.read_parquet(root / "speakers.parquet")[["speaker_name", "area", "party"]]
    sp = pd.read_parquet(root / "speeches.parquet")[["speech_id", "kind"]]
    d = sent.merge(hyp[["sentence_id", "greedy", "lm"]], on="sentence_id").merge(spk, on="speaker_name", how="left")
    d = d.merge(sp, on="speech_id", how="left")
    d = d[d["area"].notna() & (d["kind"] != "procedural")]
    excluded = d["is_nn"] | d["ref_raw"].str.contains(r"\d")
    print(f"excluded {excluded.sum()} of {len(d)} sentences: Nynorsk-transcribed (translated reference) or digits in reference")
    d = d[~excluded]
    fold = lambda s: s.replace("é", "e").replace("è", "e").replace("ò", "o")
    d = d.assign(ref=d["ref"].map(fold), greedy=d["greedy"].map(fold), lm=d["lm"].map(fold))
    rows = []
    for r in d.itertuples():
        for dec in ("greedy", "lm"):
            for k, o in outcomes(r.ref, getattr(r, dec)):
                rows.append({"decoder": dec, "speaker_name": r.speaker_name, "area": r.area, "party": r.party,
                             "speech_kind": r.kind, "word_kind": k, "outcome": o, "is_nn": r.is_nn})
    t = pd.DataFrame(rows)
    t.to_parquet(out / "critical_words.parquet")
    t["lost"] = t["outcome"].isin(["dropped", "substituted"])
    rng = np.random.default_rng(0)
    res = []
    for (dec, wk), g in t.groupby(["decoder", "word_kind"]):
        for area in AREAS:
            ga = g[g["area"] == area]
            s = ga.groupby("speaker_name").agg(n=("lost", "size"), lost=("lost", "sum"), drop=("outcome", lambda x: (x == "dropped").sum()),
                                               sub=("outcome", lambda x: (x == "substituted").sum()))
            a = s[["lost", "n"]].to_numpy(float)
            idx = rng.integers(0, len(a), (B, len(a)))
            bs = a[idx].sum(1)
            lo, hi = np.percentile(bs[:, 0] / bs[:, 1], [2.5, 97.5])
            res.append({"decoder": dec, "word_kind": wk, "area": area, "speakers": len(s), "tokens": int(s["n"].sum()),
                        "lost": s["lost"].sum() / s["n"].sum(), "lost_lo": lo, "lost_hi": hi,
                        "dropped": s["drop"].sum() / s["n"].sum(), "substituted": s["sub"].sum() / s["n"].sum()})
    res = pd.DataFrame(res)
    res.to_csv(out / "critical_words.csv", index=False)
    for (dec, wk), g in res.groupby(["decoder", "word_kind"]):
        print(f"\n{wk}s, {dec}: lost (dropped + substituted) % [95% CI over speakers]")
        for r in g.itertuples():
            print(f"  {r.area:6s} {r.speakers:3d} speakers {r.tokens:5d} tokens  lost {100 * r.lost:5.1f} [{100 * r.lost_lo:4.1f},{100 * r.lost_hi:4.1f}]"
                  f"  dropped {100 * r.dropped:4.1f}  substituted {100 * r.substituted:4.1f}")
    v = t[(t["word_kind"] == "negation") & (t["decoder"] == "lm")]["outcome"].value_counts().to_dict()
    print("\nnegation outcomes (lm):", v)

if __name__ == "__main__":
    main()
