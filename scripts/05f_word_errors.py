import json

import jiwer
import numpy as np
import pandas as pd
import statsmodels.api as sm

from dialect_probing.config import load_config
from dialect_probing.step5 import Activations, unit_rows

MIN_EAST_SPEAKERS = 2

def token_errors(cfg: dict, words: pd.DataFrame) -> pd.Series:
    utt = pd.read_parquet(cfg["out_dir"] / "align" / "utterances.parquet").set_index("utt_id")
    pos = words.sort_values("word_idx").groupby("utt_id").cumcount()
    err = pd.Series(np.nan, index=words.index)
    for uid, g in words.groupby("utt_id"):
        ref, hyp = utt.loc[uid, "ref"], utt.loc[uid, "hyp"]
        n = len(ref.split())
        if n != len(g):
            continue
        ok = np.zeros(n, dtype=bool)
        for c in jiwer.process_words(ref, hyp if hyp.strip() else "<empty>").alignments[0]:
            if c.type == "equal":
                ok[c.ref_start_idx:c.ref_end_idx] = True
        err[g.index] = (~ok[pos[g.index].to_numpy()]).astype(float)
    return err

def centroid_distances(x: np.ndarray, meta: pd.DataFrame, east: list[str]) -> tuple[np.ndarray, np.ndarray]:
    x = unit_rows(x)
    d_east = np.full(len(x), np.nan)
    d_all = np.full(len(x), np.nan)
    is_east = meta["group"].isin(east).to_numpy()
    spk = meta["speaker"].to_numpy()
    for _, idx in meta.groupby("word").indices.items():
        xs, sp, ie = x[idx], spk[idx], is_east[idx]
        tot, tot_e = xs.sum(0), xs[ie].sum(0)
        n, n_e = len(idx), ie.sum()
        for s in np.unique(sp):
            m = sp == s
            own = xs[m].sum(0)
            c_all = (tot - own) / max(n - m.sum(), 1)
            own_e = own if ie[m][0] else 0
            k_e = n_e - (m.sum() if ie[m][0] else 0)
            others_e = len(np.unique(sp[ie & ~m]))
            d_all[idx[m]] = 1 - xs[m] @ c_all / np.linalg.norm(c_all)
            if others_e >= MIN_EAST_SPEAKERS and k_e > 0:
                c_e = (tot_e - own_e) / k_e
                d_east[idx[m]] = 1 - xs[m] @ c_e / np.linalg.norm(c_e)
    return d_east, d_all

def fit(df: pd.DataFrame, cols: list[str]) -> dict:
    z = df[cols].apply(lambda c: (c - c.mean()) / c.std())
    fe = pd.get_dummies(df[["word", "speaker"]], drop_first=True, dtype=float)
    X = sm.add_constant(pd.concat([z, fe], axis=1))
    res = sm.GLM(df["err"].astype(float), X, family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(df["speaker"])[0]}, maxiter=200)
    out = {}
    for c in cols:
        out[f"b_{c}"] = res.params[c]
        out[f"se_{c}"] = res.bse[c]
        out[f"p_{c}"] = res.pvalues[c]
    return out

def within_type(df: pd.DataFrame, col: str) -> tuple[float, float]:
    diffs = []
    for _, g in df.groupby("word"):
        e = g["err"].astype(bool)
        if e.sum() >= 2 and (~e).sum() >= 2:
            diffs.append(g.loc[e, col].mean() - g.loc[~e, col].mean())
    diffs = np.array(diffs)
    return float(np.mean(diffs > 0)), len(diffs)

def main():
    cfg = load_config()
    out = cfg["out_dir"] / "step5"
    east = cfg["analysis"]["east_groups"]
    act = Activations(cfg["out_dir"] / "activations")
    lags = json.loads((out / "selected_lags.json").read_text())
    half = pd.read_csv(out / "split.csv").set_index("speaker")["half"]

    w = act.words.assign(half=act.words["speaker"].map(half))
    w = w[(w["half"] == "analysis") & ~w["group"].isin(cfg["analysis"]["exclude_groups"])]
    w = w.assign(err=token_errors(cfg, w))
    keep = (w["written_standard"] == "bokmål") & w["err"].notna()
    print(f"analysis half, Bokmål prompts: {keep.sum()} tokens, token error rate {w.loc[keep, 'err'].mean():.3f}")
    rows = w.index.to_numpy()
    meta = w.reset_index(drop=True)
    keep = keep.to_numpy()

    res, tokens = [], []
    for li, name in enumerate(act.layers):
        d_east, d_all = centroid_distances(act.vectors(rows, li, "lag", lags[name]), meta, east)
        df = meta.assign(d_east=d_east, d_all=d_all)[keep].dropna(subset=["d_east", "d_all"])
        var = df.groupby("word")["err"].transform(lambda s: 0 < s.mean() < 1)
        df = df[var.astype(bool)]
        r = {"layer": name, "layer_idx": li, "tokens": len(df), "types": df["word"].nunique(),
             "error_rate": df["err"].mean(), "corr_east_all": df[["d_east", "d_all"]].corr().iloc[0, 1]}
        r.update(fit(df, ["d_east", "d_all"]))
        r.update({f"{k}_alone": v for k, v in fit(df, ["d_east"]).items()})
        ne = df[~df["group"].isin(east)]
        r.update({f"{k}_non_east": v for k, v in fit(ne, ["d_east", "d_all"]).items()})
        df = df.assign(east_resid=df["d_east"] - np.poly1d(np.polyfit(df["d_all"], df["d_east"], 1))(df["d_all"]))
        r["within_frac_farther"], r["within_types"] = within_type(df, "east_resid")
        res.append(r)
        print(f"{name:9s} n {r['tokens']} ({r['types']} types)  r(east,all) {r['corr_east_all']:.2f}  "
              f"east {r['b_d_east']:+.3f} (p {r['p_d_east']:.3g})  all {r['b_d_all']:+.3f} (p {r['p_d_all']:.3g})  "
              f"east alone {r['b_d_east_alone']:+.3f}  non East {r['b_d_east_non_east']:+.3f} (p {r['p_d_east_non_east']:.3g})  "
              f"within type {r['within_frac_farther']:.2f} of {r['within_types']}", flush=True)
        if name == "L8":
            tokens = df[["utt_id", "word_idx", "word", "speaker", "group", "err", "d_east", "d_all"]]
    pd.DataFrame(res).to_csv(out / "word_errors.csv", index=False)
    tokens.to_parquet(out / "word_errors_tokens_L8.parquet")

    lw = json.loads((cfg["out_dir"] / "layer_weights.json").read_text())["weights"]
    probe = pd.read_csv(out / "dialect_probe.csv")
    conv = pd.read_csv(out / "convergence.csv").query("variant == 'lag' and subset == 'all'")
    t = pd.DataFrame({"layer": act.layers[:13], "mix_weight": lw})
    t = t.merge(probe[["layer", "lag"]].rename(columns={"lag": "dialect_probe"}), on="layer")
    t = t.merge(conv[["layer", "gap"]], on="layer")
    t = t.merge(pd.DataFrame(res)[["layer", "b_d_east"]], on="layer")
    t.to_csv(out / "mix_vs_dialect.csv", index=False)
    print("\nlayer mix weights vs dialect signal:")
    print(t.round(3).to_string(index=False))
    print(f"mix weight in L7-L9: {sum(lw[7:10]):.3f}  (uniform {3 / 13:.3f})")

if __name__ == "__main__":
    main()
