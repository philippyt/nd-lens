import collections

import jiwer
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from norsk_asr.text import normalize
from scipy.stats import spearmanr

from dialect_probing.config import ROOT, load_config

WORDS = ["så", "og", "jeg", "ja", "jo", "da", "egentlig", "liksom", "du", "man", "dem", "de"]
BEFORE_SA = ["ellers", "sommer", "dag", "da", "år", "der", "det", "ikke", "og", "men"]
VERBS = {"har", "er", "var", "ble", "kan", "skal", "vil", "må", "gikk", "fikk", "tok", "kom", "hadde", "blir", "gjorde",
         "begynte", "dro", "reiste", "flytta", "flyttet"}

def counts(docs: list[list[str]]):
    n = sum(map(len, docs))
    uni = collections.Counter(w for d in docs for w in d)
    bi = collections.Counter(p for d in docs for p in zip(d, d[1:]))
    res = sum(d[i] == "så" and d[i - 1] not in {"og", "men", "eller"} and d[i + 1] in VERBS for d in docs for i in range(1, len(d) - 1))
    return n, uni, bi, res

def main():
    cfg = load_config()
    tr = pd.concat([pd.read_parquet(p, columns=["text", "kept"]) for p in sorted((cfg["data_dir"] / "train_index").glob("*.parquet"))])
    seg = pd.read_parquet(cfg["data_dir"] / "spont" / "segments.parquet").query("set == 'nbtale_m3'")
    sets = {"stortinget_train": [normalize(t).split() for t in tr.loc[tr["kept"], "text"]], "module3_ref": [r.split() for r in seg["ref"]]}
    c = {k: counts(v) for k, v in sets.items()}
    rows = [{"item": w, **{k: 1e3 * u[w] / n for k, (n, u, _, _) in c.items()}} for w in WORDS]
    rows += [{"item": f"{a} så", **{k: 1e3 * b[(a, "så")] / n for k, (n, _, b, _) in c.items()}} for a in BEFORE_SA]
    rows.append({"item": "X så VERB (resumptive)", **{k: 1e3 * r / n for k, (n, _, _, r) in c.items()}})
    d = pd.DataFrame(rows)
    d["ratio"] = d["module3_ref"] / d["stortinget_train"]
    d.to_csv(ROOT / "results" / "train_conventions.csv", index=False)
    print({k: n for k, (n, *_) in c.items()}, "words")
    print("per 1000 words:")
    print(d.round(3).to_string(index=False))

    # per word type: are words that are rare in training (relative to module 3) deleted and substituted more?
    (nt, ut, _, _), (nm, um, _, _) = c["stortinget_train"], c["module3_ref"]
    hyp = pd.read_parquet(cfg["out_dir"] / "weight_space" / "tune_hyps.parquet")
    occ, dele, sub = collections.Counter(), collections.Counter(), collections.Counter()
    for r in hyp.itertuples():
        ref = r.ref.split()
        for a in jiwer.process_words(r.ref, r.hyp or "<empty>").alignments[0]:
            for w in ref[a.ref_start_idx:a.ref_end_idx]:
                occ[w] += 1
                dele[w] += a.type == "delete"
                sub[w] += a.type == "substitute"
    w = pd.DataFrame([{"word": x, "n": n, "del_rate": dele[x] / n, "sub_rate": sub[x] / n, "chars": len(x),
                       "log_underexposure": np.log((um[x] / nm) / ((ut[x] + 1) / nt))} for x, n in occ.items() if n >= 20])
    w["log_n"] = np.log(w["n"])
    w.sort_values("log_underexposure", ascending=False).to_csv(ROOT / "results" / "word_underexposure.csv", index=False)
    print(f"\n{len(w)} word types with >= 20 tune-half occurrences (greedy, unedited)")
    for col in ("del_rate", "sub_rate"):
        r = spearmanr(w[col], w["log_underexposure"])
        fit = smf.wls(f"{col} ~ log_underexposure + chars + log_n", w, weights=w["n"]).fit()
        lo, hi = fit.conf_int().loc["log_underexposure"]
        print(f"{col}: spearman {r.statistic:+.2f} (p {r.pvalue:.1g}); weighted slope per log unit "
              f"{fit.params['log_underexposure']:+.3f} [{lo:+.3f}, {hi:+.3f}] controlling for length and frequency")
    print(w.sort_values("log_underexposure", ascending=False).head(10).round(3).to_string(index=False))

if __name__ == "__main__":
    main()
