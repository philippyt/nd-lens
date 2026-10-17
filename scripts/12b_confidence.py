import numpy as np
import pandas as pd
from norsk_asr.lm import ctc_labels, make_decoder
from norsk_asr.units import Units
from scipy.stats import rankdata

from dialect_probing.config import ROOT, load_config
from dialect_probing.m3_decode import align, expand, gap_scores, greedy_words, lm_decode, load_logprobs, word_confidence

NB = 2000
BUDGETS = (0.10, 0.20)

def auroc(score: np.ndarray, y: np.ndarray) -> float:
    n1 = y.sum()
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    return (rankdata(score)[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

def rows_for(dec: str, words: list, ix: pd.DataFrame, lps: list) -> tuple[list, list]:
    wr, gr = [], []
    for i, (r, w) in enumerate(zip(ix.itertuples(), words)):
        toks, spans = expand(w)
        lab, _, dels = align(r.ref, toks)
        conf = [word_confidence(lps[i], sp) for sp in spans]
        tot, peak = gap_scores(lps[i], spans)
        meta = {"decoder": dec, "seg": i, "speaker": r.speaker, "area": r.area, "half": r.half}
        for k, (c, y) in enumerate(zip(conf, lab)):
            wr.append({**meta, "k": k, "conf_mean": c[0], "conf_min": c[1], "wrong": int(y)})
        for k in range(len(tot)):
            gr.append({**meta, "k": k, "gap_total": tot[k], "gap_peak": peak[k], "dels": int(dels[k])})
    return wr, gr

def metrics(w: pd.DataFrame, g: pd.DataFrame, wcol: str, gcol: str, wthr: dict, gthr: dict) -> dict:
    y = w["wrong"].to_numpy()
    s = -w[wcol].to_numpy()
    gd = g["dels"].to_numpy()
    gs = g[gcol].to_numpy()
    out = {"words": len(w), "si_errors": int(y.sum()), "deletions": int(gd.sum()),
           "auroc_words": auroc(s, y), "auroc_gaps": auroc(gs, (gd > 0).astype(int))}
    nerr = y.sum() + gd.sum()
    for b in BUDGETS:
        fw = w[wcol].to_numpy() <= wthr[b]
        fg = gs >= gthr[b]
        cw, cg = (fw & (y == 1)).sum(), gd[fg].sum()
        tag = f"{int(b * 100)}"
        out |= {f"flagged_words_{tag}": fw.mean(), f"caught_si_{tag}": cw / max(y.sum(), 1),
                f"precision_{tag}": cw / max(fw.sum(), 1), f"caught_del_{tag}": cg / max(gd.sum(), 1),
                f"caught_all_{tag}": (cw + cg) / max(nerr, 1)}
        adj = np.zeros(len(gd), bool)
        flag_by_seg = w.assign(f=fw).groupby("seg")["f"].apply(np.asarray).to_dict()
        for j, (sg, k) in enumerate(zip(g["seg"].to_numpy(), g["k"].to_numpy())):
            f = flag_by_seg.get(sg)
            if f is not None and gd[j]:
                adj[j] = (k > 0 and f[k - 1]) or (k < len(f) and f[k])
        out[f"del_next_to_flag_{tag}"] = gd[adj].sum() / max(gd.sum(), 1)
    return out

def by_speaker(df: pd.DataFrame) -> dict:
    return {s: d for s, d in df.groupby("speaker")}

def main():
    cfg = load_config()
    out = cfg["out_dir"] / "spont"
    units = Units.load(cfg["checkpoint"] / "units.json")
    ix = pd.read_parquet(out / "m3_logprobs_index.parquet").merge(pd.read_csv(out / "m3_split.csv")[["speaker", "half"]], on="speaker")
    lps = load_logprobs(out / "m3_logprobs.npz")
    dec = make_decoder(str(cfg["norsk_asr_dir"] / "data" / "lm" / "no4"), ctc_labels(units), 0.6, 1.0)
    hyps = {"greedy": [greedy_words(lp, units.pieces) for lp in lps], "lm": lm_decode(dec, lps, list(range(len(lps))))}
    W, G = [], []
    for name, words in hyps.items():
        wr, gr = rows_for(name, words, ix, lps)
        W += wr
        G += gr
    W, G = pd.DataFrame(W), pd.DataFrame(G)
    W.to_parquet(out / "m3_conf_words.parquet")
    G.to_parquet(out / "m3_conf_gaps.parquet")
    res, diffs = [], []
    rng = np.random.default_rng(0)
    for name in hyps:
        w, g = W[W["decoder"] == name], G[G["decoder"] == name]
        wt, gt = w[w["half"] == "tune"], g[g["half"] == "tune"]
        wcol = max(("conf_mean", "conf_min"), key=lambda c: auroc(-wt[c].to_numpy(), wt["wrong"].to_numpy()))
        gcol = max(("gap_total", "gap_peak"), key=lambda c: auroc(gt[c].to_numpy(), (gt["dels"] > 0).astype(int).to_numpy()))
        wthr = {b: np.quantile(wt[wcol], b) for b in BUDGETS}
        gthr = {b: np.quantile(gt[gcol], 1 - b) for b in BUDGETS}
        print(f"{name}: word score {wcol}, gap score {gcol} (chosen on tune); thresholds {', '.join(f'{b:.0%}: {wthr[b]:.2f}' for b in BUDGETS)}")
        we, ge = w[w["half"] == "test"], g[g["half"] == "test"]
        for area in ["all", *sorted(we["area"].unique())]:
            a = (lambda d: d) if area == "all" else (lambda d, area=area: d[d["area"] == area])
            res.append({"decoder": name, "area": area, "speakers": a(we)["speaker"].nunique(), **metrics(a(we), a(ge), wcol, gcol, wthr, gthr)})
        sw, se = by_speaker(we[we["area"] == "west"]), by_speaker(we[we["area"] == "east"])
        gw, gE = by_speaker(ge[ge["area"] == "west"]), by_speaker(ge[ge["area"] == "east"])
        boots = []
        for _ in range(NB):
            m = []
            for sp, gp in ((sw, gw), (se, gE)):
                keys = list(sp)
                pick = [keys[j] for j in rng.integers(0, len(keys), len(keys))]
                ww = pd.concat([sp[k].assign(seg=sp[k]["seg"] + n * 100000) for n, k in enumerate(pick)])
                gg = pd.concat([gp[k].assign(seg=gp[k]["seg"] + n * 100000) for n, k in enumerate(pick)])
                m.append(metrics(ww, gg, wcol, gcol, wthr, gthr))
            boots.append({k: m[0][k] - m[1][k] for k in m[0]})
        boots = pd.DataFrame(boots)
        rw, re = (next(r for r in res if r["decoder"] == name and r["area"] == a) for a in ("west", "east"))
        for k in [c for c in boots.columns if c.startswith(("auroc", "caught", "precision", "flagged", "del_next"))]:
            lo, hi = np.nanpercentile(boots[k], [2.5, 97.5])
            diffs.append({"decoder": name, "metric": k, "west": rw[k], "east": re[k], "diff": rw[k] - re[k], "lo": lo, "hi": hi})
    res, diffs = pd.DataFrame(res), pd.DataFrame(diffs)
    res.to_csv(ROOT / "results" / "confidence_areas.csv", index=False)
    diffs.to_csv(ROOT / "results" / "confidence_west_east.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 40)
    cols = ["decoder", "area", "speakers", "words", "si_errors", "deletions", "auroc_words", "auroc_gaps", "flagged_words_10",
            "caught_si_10", "caught_si_20", "caught_del_10", "caught_del_20", "del_next_to_flag_20", "caught_all_10", "caught_all_20"]
    print("\ntest half\n" + res[cols].round(3).to_string(index=False))
    print(f"\nWest minus East, test half, 95% CI over speakers ({NB} resamples)\n" + diffs.round(3).to_string(index=False))

if __name__ == "__main__":
    main()
