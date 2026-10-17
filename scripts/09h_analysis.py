import json
import re

import jiwer
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
from statsmodels.stats.proportion import proportion_confint

from dialect_probing.config import ROOT, load_config

FIELDS = ["problem", "fakta", "standpunkt", "tiltak"]
CONFIDENT = {"feilhort", "tillegg", "feilattribusjon"}
AREAS = ["east", "west", "mid", "north", "south"]
STOP = {"som", "det", "har", "ikke", "eller", "også", "skal", "kan", "være", "vært", "blir", "ble", "denne", "dette",
        "disse", "andre", "mange", "mye", "noen", "noe", "hvor", "hvordan", "fordi", "når", "etter", "under", "over",
        "mellom", "gjennom", "president", "representanten", "statsråden", "taleren", "ikke nevnt", "nevnt"}
B = 10000

def boot_rate(df: pd.DataFrame, col: str, rng) -> tuple[float, float, float]:
    s = df.groupby("speaker_name")[col].agg(["sum", "size"]).to_numpy(float)
    idx = rng.integers(0, len(s), (B, len(s)))
    b = s[idx].sum(1)
    r = b[:, 0] / b[:, 1]
    return s[:, 0].sum() / s[:, 1].sum(), *np.percentile(r, [2.5, 97.5])

def boot_diff(a: pd.DataFrame, b: pd.DataFrame, col: str, rng) -> tuple[float, float, float]:
    sa = a.groupby("speaker_name")[col].agg(["sum", "size"]).to_numpy(float)
    sb = b.groupby("speaker_name")[col].agg(["sum", "size"]).to_numpy(float)
    ba = sa[rng.integers(0, len(sa), (B, len(sa)))].sum(1)
    bb = sb[rng.integers(0, len(sb), (B, len(sb)))].sum(1)
    d = ba[:, 0] / ba[:, 1] - bb[:, 0] / bb[:, 1]
    return sa[:, 0].sum() / sa[:, 1].sum() - sb[:, 0].sum() / sb[:, 1].sum(), *np.percentile(d, [2.5, 97.5])

def kappa(x: np.ndarray, y: np.ndarray) -> float:
    po = (x == y).mean()
    pe = x.mean() * y.mean() + (1 - x.mean()) * (1 - y.mean())
    return (po - pe) / (1 - pe) if pe < 1 else 1.0

def flat(v) -> str:
    return " ".join(flat(x) for x in v) if isinstance(v, list) else str(v)

def tokens(v) -> set[str]:
    text = flat(v)
    return {w for w in re.findall(r"[a-zæøåé]+", text.lower()) if len(w) >= 4 and w not in STOP}

def lost_words(ref: str, hyp: str) -> set[str]:
    r = ref.split()
    lost = set()
    for c in jiwer.process_words(ref, hyp or "<empty>").alignments[0]:
        if c.type in ("delete", "substitute"):
            lost |= {w for w in r[c.ref_start_idx:c.ref_end_idx] if len(w) >= 4 and w not in STOP}
    return lost

def mixed(df: pd.DataFrame, formula: str) -> pd.DataFrame:
    m = BinomialBayesMixedGLM.from_formula(formula, {"speaker": "0 + C(speaker_name)"}, df)
    r = m.fit_vb()
    t = pd.DataFrame({"coef": r.fe_mean, "sd": r.fe_sd}, index=m.exog_names)
    t["lo"], t["hi"] = t["coef"] - 1.96 * t["sd"], t["coef"] + 1.96 * t["sd"]
    return t[t.index.str.contains("area")]

def main():
    cfg = load_config()
    out = cfg["out_dir"] / "npsc"
    sample = pd.read_parquet(out / "sample.parquet")
    spk = pd.read_parquet(cfg["data_dir"] / "npsc" / "speakers.parquet")[["speaker_name", "dialect_source"]]
    sample = sample.merge(spk, on="speaker_name", how="left")
    sample["party"] = sample["party"].fillna("ukjent")
    notes = {(r["speech_id"], r["run"]): r["note"] for r in map(json.loads, open(out / "notes.jsonl", encoding="utf-8"))}
    judge = {(r["speech_id"], r["pair"]): r["labels"] for r in map(json.loads, open(out / "judge.jsonl", encoding="utf-8"))}

    rows, dirs = [], []
    for s in sample.itertuples():
        p1, p2 = judge.get((s.speech_id, "ref1-asr")), judge.get((s.speech_id, "ref1-ref2"))
        if not p1 or not p2:
            continue
        for f in FIELDS:
            rows.append({"speech_id": s.speech_id, "speaker_name": s.speaker_name, "area": s.area, "party": s.party,
                         "dialect_source": s.dialect_source, "nn": s.nn, "wer": s.wer, "field": f,
                         "asr": p1["asr"][f], "ref1_p1": p1["ref1"][f], "ref1_p2": p2["ref1"][f], "ref2": p2["ref2"][f]})
        dirs.append({"speech_id": s.speech_id, "speaker_name": s.speaker_name, "area": s.area, "asr": p1["asr"]["retning"],
                     "ref1_p1": p1["ref1"]["retning"], "ref1_p2": p2["ref1"]["retning"], "ref2": p2["ref2"]["retning"]})
    d = pd.DataFrame(rows)
    dd = pd.DataFrame(dirs)
    ok = lambda c: d[c] == "korrekt"
    d["asr_err"] = (~ok("asr")).astype(int)
    d["ref2_err"] = (~ok("ref2")).astype(int)
    d["asr_caused"] = ((~ok("asr")) & ok("ref1_p1") & ok("ref1_p2") & ok("ref2")).astype(int)
    d["asr_caused_confident"] = (d["asr_caused"].astype(bool) & d["asr"].isin(CONFIDENT)).astype(int)
    d["excess"] = d["asr_err"] - d["ref2_err"]
    d.to_parquet(out / "b_fields.parquet")
    print(f"speeches judged {d['speech_id'].nunique()} of {len(sample)}; speech-fields {len(d)}")

    print("\njudge consistency, ref1 graded twice (error vs korrekt):")
    for f in FIELDS:
        g = d[d["field"] == f]
        print(f"  {f:10s} agreement {(ok('ref1_p1')[g.index] == ok('ref1_p2')[g.index]).mean():.2f}  "
              f"kappa {kappa(ok('ref1_p1')[g.index].to_numpy(), ok('ref1_p2')[g.index].to_numpy()):.2f}")

    rng = np.random.default_rng(0)
    res = []
    for area in AREAS + ["all"]:
        g = d if area == "all" else d[d["area"] == area]
        r = {"area": area, "speakers": g["speaker_name"].nunique(), "speeches": g["speech_id"].nunique()}
        for col in ("asr_err", "ref2_err", "asr_caused", "asr_caused_confident"):
            r[col], r[f"{col}_lo"], r[f"{col}_hi"] = boot_rate(g, col, rng)
        r["excess"], r["excess_lo"], r["excess_hi"] = boot_rate(g.assign(x=g["excess"]), "x", rng)
        res.append(r)
    res = pd.DataFrame(res)
    res.to_csv(out / "b_areas.csv", index=False)
    pct = lambda r, c: f"{100 * r[c]:5.1f} [{100 * r[c + '_lo']:4.1f},{100 * r[c + '_hi']:4.1f}]"
    print("\nper area, % of speech-fields [95% CI over speakers]  (South descriptive only)")
    print("  area   spk  asr error          ref2 error         excess             ASR-caused         ASR-caused confident")
    for r in res.to_dict("records"):
        print(f"  {r['area']:6s} {r['speakers']:3d}  {pct(r, 'asr_err')}  {pct(r, 'ref2_err')}  {pct(r, 'excess')}  "
              f"{pct(r, 'asr_caused')}  {pct(r, 'asr_caused_confident')}")
    w, e = d[d["area"] == "west"], d[d["area"] == "east"]
    for col in ("asr_caused", "asr_caused_confident", "excess"):
        v, lo, hi = boot_diff(w, e, col, rng)
        print(f"West minus East {col}: {100 * v:+.1f} points [{100 * lo:+.1f}, {100 * hi:+.1f}]")

    print("\nASR-caused errors by label (all areas):", d[d["asr_caused"] == 1]["asr"].value_counts().to_dict())
    print("ASR-caused rate by field:", d.groupby("field")["asr_caused"].mean().round(3).to_dict())

    print("\nmixed logistic, ASR-caused ~ area + field (+ party), speaker random effect (log-odds vs East):")
    base = "asr_caused ~ C(area, Treatment('east')) + C(field)"
    for name, formula, data in [("without party", base, d), ("with party", base + " + C(party)", d),
                                ("with WER", base + " + wer_z", d.assign(wer_z=(d["wer"] - d["wer"].mean()) / d["wer"].std())),
                                ("no county-only labels", base, d[~d["dialect_source"].isin(["birth_county", "rep_county"])])]:
        t = mixed(data, formula)
        t.index = t.index.str.extract(r"T\.(\w+)")[0].fillna(t.index.to_series()).to_numpy()
        print(f"  {name}: " + "  ".join(f"{i} {r.coef:+.2f} [{r.lo:+.2f},{r.hi:+.2f}]" for i, r in t.iterrows()))

    print("\nNynorsk-origin speeches (NPSC transcribed in Nynorsk) vs Bokmål-origin, ASR-caused %:")
    for lab, g in d.groupby(d["nn"] > 0.5):
        v, lo, hi = boot_rate(g, "asr_caused", rng)
        print(f"  {'nynorsk' if lab else 'bokmål':8s} speeches {g['speech_id'].nunique():3d}  {100 * v:.1f} [{100 * lo:.1f},{100 * hi:.1f}]")
    sens = d[~d["dialect_source"].isin(["birth_county", "rep_county"])]
    print(f"\nwithout county-only labels ({d['speaker_name'].nunique() - sens['speaker_name'].nunique()} speakers dropped):")
    for area in AREAS:
        g = sens[sens["area"] == area]
        v, lo, hi = boot_rate(g, "asr_caused", rng)
        print(f"  {area:6s} {g['speaker_name'].nunique():3d} speakers  ASR-caused {100 * v:.1f} [{100 * lo:.1f},{100 * hi:.1f}]")

    ref_agree = (dd["ref1_p1"] == dd["ref2"]) & (dd["ref1_p1"] == dd["ref1_p2"])
    polar = {"for", "mot"}
    dd["flip"] = ref_agree & dd["ref1_p1"].isin(polar) & dd["asr"].isin(polar) & (dd["asr"] != dd["ref1_p1"])
    dd["hardened"] = ref_agree & dd["ref1_p1"].isin({"spor", "blandet"}) & dd["asr"].isin(polar)
    dd["softened"] = ref_agree & dd["ref1_p1"].isin(polar) & dd["asr"].isin({"spor", "blandet"})
    dd["ref_flip"] = (dd["ref1_p1"].isin(polar) & dd["ref2"].isin(polar) & (dd["ref1_p1"] != dd["ref2"]))
    dd.to_parquet(out / "b_directions.parquet")
    print("\nstandpoint changes, count / speeches [Wilson 95% CI]:")
    for area in AREAS + ["all"]:
        g = dd if area == "all" else dd[dd["area"] == area]
        parts = []
        for c in ("flip", "hardened", "softened", "ref_flip"):
            k, n = int(g[c].sum()), len(g)
            lo, hi = proportion_confint(k, n, method="wilson")
            parts.append(f"{c} {k}/{n} [{100 * lo:.1f},{100 * hi:.1f}%]")
        print(f"  {area:6s} " + "  ".join(parts))
    for r in dd[dd["flip"] | dd["hardened"]].head(6).itertuples():
        print(f"\n  speech {r.speech_id} ({r.area}): ref {r.ref1_p1} -> asr {r.asr}")
        print("    ref1:", notes[(r.speech_id, "ref1")]["standpunkt"][:300])
        print("    asr :", notes[(r.speech_id, "asr")]["standpunkt"][:300])

    corr = []
    for s in sample.itertuples():
        if (s.speech_id, "asr") not in notes:
            continue
        lost = lost_words(s.reference, s.asr)
        in_ref = {w for w in lost if w in tokens(list(notes[(s.speech_id, "ref1")].values()))}
        in_asr = {w for w in in_ref if w in tokens(list(notes[(s.speech_id, "asr")].values()))}
        corr.append({"area": s.area, "speaker_name": s.speaker_name, "lost_in_note": len(in_ref), "corrected": len(in_asr)})
    c = pd.DataFrame(corr)
    print("\ncorrected: content words lost by ASR but needed by the reference note")
    for area, g in c.groupby("area"):
        print(f"  {area:6s} lost {g['lost_in_note'].sum():4d}  corrected {g['corrected'].sum():4d} ({g['corrected'].sum() / max(g['lost_in_note'].sum(), 1):.0%})")

    style = {"figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb", "axes.edgecolor": "#e4e3df",
             "axes.labelcolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.spines.top": False,
             "axes.spines.right": False, "axes.grid": True, "grid.color": "#e4e3df", "axes.axisbelow": True,
             "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlesize": 12, "legend.frameon": False}
    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        r = res.set_index("area").loc[AREAS]
        for k, (col, label, color) in enumerate([("asr_caused", "ASR-caused", "#2a78d6"), ("asr_caused_confident", "Confidently wrong", "#eb6834")]):
            x = np.arange(len(AREAS)) + (k - 0.5) * 0.18
            ax.errorbar(x, r[col] * 100, yerr=[(r[col] - r[col + "_lo"]) * 100, (r[col + "_hi"] - r[col]) * 100],
                        fmt="o", color=color, elinewidth=1.5, capsize=0, label=label, markersize=7)
        ax.set_xticks(range(len(AREAS)), [f"{a.capitalize()}\n{int(r.loc[a, 'speakers'])} speakers" for a in AREAS])
        ax.set_ylabel("% of note fields")
        ax.set_title("Recognition errors in notes by area")
        ax.set_ylim(bottom=-3)
        ax.set_yticks(range(0, 20, 5))
        ax.legend(loc="lower left", ncol=2)
        fig.tight_layout()
        fig.savefig(ROOT / "results" / "note_errors_areas.png", dpi=160)

if __name__ == "__main__":
    main()
