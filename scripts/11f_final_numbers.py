from collections import defaultdict

import numpy as np
import pandas as pd

from dialect_probing.config import load_config
from dialect_probing.written_standard import LEXICONS

CROSS = {("sjøl", "selv"), ("heilt", "helt"), ("heile", "hele"), ("òg", "også"), ("ifra", "fra"), ("sia", "siden"),
         ("anna", "annet"), ("trondhjem", "trondheim"), ("ikkje", "ikke"), ("noko", "noe"), ("mykje", "mye"),
         ("berre", "bare"), ("heim", "hjem"), ("heime", "hjemme"), ("meir", "mer"), ("kva", "hva"), ("frå", "fra"),
         ("no", "nå"), ("dei", "de"), ("ho", "hun"), ("eg", "jeg"), ("ein", "en"), ("eit", "et"), ("vore", "vært"),
         ("vart", "ble"), ("sjå", "se"), ("kjem", "kommer"), ("korleis", "hvordan"), ("kor", "hvor")}
FUNC_POS = {"pron", "prep", "konj", "sbu", "det", "inf-merke", "interj"}
FUNC_EXTRA = {"jeg", "er", "det", "en", "ja", "så", "nå", "har", "når", "hadde", "var", "ha", "skal", "skulle", "vil",
              "ville", "kan", "kunne", "må", "blir", "bli", "ble", "blitt", "vært", "være", "da", "jo", "liksom", "altså",
              "sånn", "litt", "veldig", "også", "bare", "ikke", "der", "her", "nok", "vel", "egentlig", "ganske", "mye",
              "noe", "hvis", "nei", "hm", "eh", "mm", "òg", "ei", "ett", "et", "sjøl", "selv", "heilt", "helt", "kanskje"}
B = 10000

def boot(df: pd.DataFrame, num: str, den: str, rng) -> tuple[float, float, float]:
    w, e = df[df["area"] == "west"], df[df["area"] == "east"]
    bw = w[[num, den]].to_numpy(float)[rng.integers(0, len(w), (B, len(w)))].sum(1)
    be = e[[num, den]].to_numpy(float)[rng.integers(0, len(e), (B, len(e)))].sum(1)
    d = bw[:, 0] / bw[:, 1] - be[:, 0] / be[:, 1]
    obs = w[num].sum() / w[den].sum() - e[num].sum() / e[den].sum()
    return obs, *np.percentile(d, [2.5, 97.5])

def main():
    cfg = load_config()
    out = cfg["out_dir"] / "spont"
    forms, tags = defaultdict(set), defaultdict(set)
    with open(cfg["data_dir"] / "ordbank" / LEXICONS["nob"], encoding="latin-1") as f:
        next(f)
        for line in f:
            c = line.split("\t")
            if len(c) > 3:
                w = c[2].lower()
                forms[w].add((c[1], c[3].replace(" normert", "").strip()))
                tags[w].add(c[3].split()[0])
    t = pd.read_parquet(out / "m3_token_outcomes.parquet")
    t["variant_ok"] = [o == "substituted" and (bool(forms.get(w, set()) & forms.get(a, set())) or (w, a) in CROSS)
                       for o, w, a in zip(t["outcome"], t["word"], t["output"])]
    t["err_tol"] = ((t["outcome"] != "correct") & ~t["variant_ok"]).astype(int)
    t["func"] = [w in FUNC_EXTRA or bool(tags.get(w, set()) & FUNC_POS) for w in t["word"]]
    ins = pd.read_parquet(out / "taxonomy_segments.parquet").query("decoder == 'norsk-asr greedy'").groupby("speaker")["insertion"].sum()
    sp = t.groupby(["area", "speaker"]).agg(words=("err", "size"), err=("err", "sum"), err_tol=("err_tol", "sum"),
                                            accepted=("variant_ok", "sum")).reset_index()
    sp["ins"] = sp["speaker"].map(ins).fillna(0)
    sp["wer_err"], sp["wer_tol"] = sp["err"] + sp["ins"], sp["err_tol"] + sp["ins"]
    rng = np.random.default_rng(0)
    print("module 3, norsk-asr greedy, West vs East, 95% CI over speakers\n")
    for label, num in (("WER (as reported)", "wer_err"), ("WER, official spelling variants accepted", "wer_tol")):
        v, lo, hi = boot(sp, num, "words", rng)
        w_, e_ = sp[sp["area"] == "west"], sp[sp["area"] == "east"]
        print(f"  {label:42s} West {100 * w_[num].sum() / w_['words'].sum():5.1f}%  East {100 * e_[num].sum() / e_['words'].sum():5.1f}%  "
              f"gap {100 * v:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]")
    acc = t[t["variant_ok"]].groupby(["word", "output"]).size().sort_values(ascending=False)
    print(f"  accepted substitutions: West {sp.loc[sp['area'] == 'west', 'accepted'].sum()}, East {sp.loc[sp['area'] == 'east', 'accepted'].sum()}; "
          f"most frequent: " + ", ".join(f"{a}={b} {k}" for (a, b), k in acc.head(12).items()))
    print()
    for label, mask in (("content words only", ~t["func"]), ("function words and fillers only", t["func"])):
        sub = t[mask].groupby(["area", "speaker"]).agg(words=("err", "size"), err=("err", "sum"), err_tol=("err_tol", "sum")).reset_index()
        for sl, num in (("", "err"), (", variants accepted", "err_tol")):
            v, lo, hi = boot(sub, num, "words", rng)
            w_, e_ = sub[sub["area"] == "west"], sub[sub["area"] == "east"]
            print(f"  {label + sl:42s} West {100 * w_[num].sum() / w_['words'].sum():5.1f}%  East {100 * e_[num].sum() / e_['words'].sum():5.1f}%  "
                  f"gap {100 * v:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]  ({t[mask & (t['area'] == 'west')].shape[0] / (t['area'] == 'west').sum():.0%} of West tokens)")
    print("\n  (content/function rates are substitutions + deletions per reference word; insertions cannot be assigned to a word)")

if __name__ == "__main__":
    main()
