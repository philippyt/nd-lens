from collections import Counter, defaultdict
from difflib import SequenceMatcher

import jiwer
import numpy as np
import pandas as pd
from norsk_asr.text import normalize

from dialect_probing.config import load_config
from dialect_probing.written_standard import LEXICONS

A2 = {"jeg", "meg", "deg", "seg", "vi", "dere", "hun", "de", "ikke", "hva", "hvor", "hvordan", "hvorfor", "hvem",
      "kommer", "se", "ble", "vært", "hjem", "bare", "noe", "noen", "mye", "mer", "hvis", "også"}
VARIANTS = {"til": "te", "fra": "frå/ifra", "nå": "no/nu", "er": "e/æ", "av": "ta/tå/åv", "med": "me/mæ",
            "det": "de/dæ", "så": "sø", "skal": "ska", "når": "kor tid/ka tid", "være": "vere/vær", "dem": "dei/døm",
            "hadde": "hadd", "har": "he/ha", "mange": "mange/mang", "hjemme": "heime", "ikke": "ikkje/ittj",
            "jeg": "e/eg/æ", "hva": "kva/ka", "hvor": "kor", "hvordan": "korleis/koss", "bare": "berre", "noe": "noko"}
B = 2000

def main():
    cfg = load_config()
    out = cfg["out_dir"] / "spont"
    seg = pd.read_parquet(cfg["data_dir"] / "spont" / "segments.parquet")
    seg = seg[(seg["set"] == "nbtale_m3") & seg["area"].isin(["west", "east"])]
    hyp = pd.read_parquet(out / "hyp_norsk-asr.parquet").query("model == 'norsk-asr greedy'")[["seg_id", "hyp"]]
    seg = seg.merge(hyp, on="seg_id")
    rows = []
    for s in seg.itertuples():
        ref, h = s.ref, normalize(s.hyp or "") or "<empty>"
        rw, hw = ref.split(), h.split()
        for c in jiwer.process_words(ref, h).alignments[0]:
            if c.type == "insert":
                continue
            for i, w in enumerate(rw[c.ref_start_idx:c.ref_end_idx]):
                if c.type == "equal":
                    o, alt = "correct", w
                elif c.type == "delete":
                    o, alt = "deleted", "<deleted>"
                else:
                    j = c.hyp_start_idx + i
                    o, alt = "substituted", hw[j] if j < c.hyp_end_idx else "<deleted>"
                rows.append({"area": s.area, "speaker": s.speaker, "word": w, "outcome": o, "output": alt})
    t = pd.DataFrame(rows)
    lem = defaultdict(set)
    with open(cfg["data_dir"] / "ordbank" / LEXICONS["nob"], encoding="latin-1") as f:
        next(f)
        for line in f:
            c = line.split("\t")
            if len(c) > 2:
                lem[c[2].lower()].add(c[1])
    t["variant"] = [o == "substituted" and bool(lem.get(w, set()) & lem.get(a, set())) for o, w, a in zip(t["outcome"], t["word"], t["output"])]
    loose = [o == "substituted" and a in lem and w[:1] == a[:1] and SequenceMatcher(None, w, a).ratio() >= 0.6
             for o, w, a in zip(t["outcome"], t["word"], t["output"])]
    t["variant_loose"] = t["variant"] | pd.Series(loose, index=t.index)
    t["err"] = (t["outcome"] != "correct").astype(int)
    t.to_parquet(out / "m3_token_outcomes.parquet")
    n = t.groupby("area").size()
    total = t[t["area"] == "west"]["err"].sum() / n["west"] - t[t["area"] == "east"]["err"].sum() / n["east"]
    g = t.groupby(["word", "area"])["err"].agg(["sum", "size"]).unstack("area", fill_value=0)
    w = pd.DataFrame({"n_west": g[("size", "west")], "n_east": g[("size", "east")],
                      "err_west": g[("sum", "west")], "err_east": g[("sum", "east")]})
    w["rate_west"] = w["err_west"] / w["n_west"].replace(0, np.nan)
    w["rate_east"] = w["err_east"] / w["n_east"].replace(0, np.nan)
    w["freq_west"] = w["n_west"] / n["west"]
    w["freq_east"] = w["n_east"] / n["east"]
    w["contribution"] = w["err_west"] / n["west"] - w["err_east"] / n["east"]
    both = (w["n_west"] > 0) & (w["n_east"] > 0)
    w["rate_part"] = np.where(both, w["freq_east"] * (w["rate_west"] - w["rate_east"]), 0.0)
    w["usage_part"] = w["contribution"] - w["rate_part"]
    w = w.sort_values("contribution", ascending=False)
    outs = t[(t["area"] == "west") & (t["err"] == 1)].groupby("word")["output"].agg(lambda s: Counter(s).most_common(4))
    w["west_outputs"] = outs.reindex(w.index).map(lambda v: ", ".join(f"{o} {k}" for o, k in v) if isinstance(v, list) else "")
    w["in_a2"] = w.index.isin(A2)
    w["dialect_variant"] = w.index.map(VARIANTS).fillna("")
    w.to_csv(out / "word_ranking.csv")

    print(f"West-East gap in substitution+deletion rate: {100 * total:+.2f} points "
          f"(West {n['west']} reference tokens, East {n['east']}); word types {len(w)}")
    print("note: a word's contribution = its West errors per West token minus its East errors per East token; "
          "it splits into a rate part (West-East error rate x East frequency) and a usage part (West uses the word more or less often)")
    print(f"  rate part (West errs more on the same word) {w['rate_part'].sum() / total:.0%}, "
          f"usage part (West says error-prone words more often) {w['usage_part'].sum() / total:.0%}")
    shared = set(w.index[(w["n_west"] >= 5) & (w["n_east"] >= 5)])
    s = t[t["word"].isin(shared)]
    ns = s.groupby("area").size()
    print(f"  restricted to {len(shared)} word types used >= 5 times in both areas ({ns['west'] / n['west']:.0%} of West tokens): "
          f"West {100 * s[s['area'] == 'west']['err'].mean():.1f}% vs East {100 * s[s['area'] == 'east']['err'].mean():.1f}%, "
          f"gap {100 * (s[s['area'] == 'west']['err'].mean() - s[s['area'] == 'east']['err'].mean()):+.2f} points")
    for kind, mask in (("deletions", t["outcome"] == "deleted"), ("substitutions to another form of the same lemma", t["variant"]),
                       ("substitutions to a similar standard Bokmål word (loose upper bound)", t["variant_loose"]),
                       ("other substitutions", (t["outcome"] == "substituted") & ~t["variant"])):
        part = t[mask & (t["area"] == "west")].shape[0] / n["west"] - t[mask & (t["area"] == "east")].shape[0] / n["east"]
        print(f"  {kind}: {100 * part:+.2f} points ({part / total:.0%} of the gap)")
    vt = t[t["variant"]].groupby(["word", "output"]).size().sort_values(ascending=False).head(15)
    print("  most frequent same-lemma substitutions (reference form -> model form):", ", ".join(f"{a}->{b} {k}" for (a, b), k in vt.items()))
    cum = w["contribution"].cumsum() / total
    for k in (10, 50, 200):
        print(f"  top {k:3d} word types explain {cum.iloc[k - 1]:.0%} of the gap")
    print(f"  word types with positive contribution {(w['contribution'] > 0).sum()}, negative {(w['contribution'] < 0).sum()}; "
          f"sum of positives {w.loc[w['contribution'] > 0, 'contribution'].sum() / total:.0%} of the gap")
    print("\ntop 40 word types by contribution to the West-East gap:")
    print(f"  {'word':12s} {'nW':>5s} {'nE':>5s} {'rateW':>6s} {'rateE':>6s} {'contrib':>7s} {'rate':>6s} {'usage':>6s}  a2  variant         West outputs instead")
    for word, r in w.head(40).iterrows():
        print(f"  {word:12s} {r.n_west:5d} {r.n_east:5d} {r.rate_west:6.2f} {r.rate_east if not np.isnan(r.rate_east) else float('nan'):6.2f} "
              f"{100 * r.contribution:+7.2f} {100 * r.rate_part:+6.2f} {100 * r.usage_part:+6.2f}  {'yes' if r.in_a2 else ' - '} "
              f"{r.dialect_variant:15s} {r.west_outputs}")
    flag = w.head(40)
    flag = flag[(flag["dialect_variant"] != "") & ~flag["in_a2"]]
    print("\nlikely dialect-variable words in the top 40 that the a2 list missed:", ", ".join(f"{i} ({v})" for i, v in flag["dialect_variant"].items()))
    for word in ["til", "fra", "nå", "er", "hva", "hvor", "hvordan", "bare", "jeg"]:
        if word in w.index:
            r = w.loc[word]
            print(f"  {word:8s} rank {w.index.get_loc(word) + 1:4d}  rateW {r.rate_west:.2f} rateE {r.rate_east:.2f}  contribution {100 * r.contribution:+.2f}  outputs {r.west_outputs}")

if __name__ == "__main__":
    main()
