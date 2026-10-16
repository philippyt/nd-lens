import json

import jiwer
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from dialect_probing.config import load_config
from dialect_probing.step5 import Activations, PairStats

def perm_p(a: np.ndarray, b: np.ndarray, stat, n: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    obs = stat(a, b)
    null = np.array([stat(a, rng.permutation(b)) for _ in range(n)])
    return float((1 + (np.abs(null) >= abs(obs)).sum()) / (1 + n))

def error_rates(cfg: dict) -> pd.DataFrame:
    hyp = pd.read_parquet(cfg["out_dir"] / "align" / "utterances.parquet")
    utt = pd.read_parquet(cfg["data_dir"] / "nbtale" / "utterances.parquet")[["utt_id", "group", "written_standard"]]
    u = hyp.merge(utt, on="utt_id")
    u = u[~u["group"].isin(cfg["analysis"]["exclude_groups"])]
    rows = []
    for (grp, ws), g in u.groupby(["group", "written_standard"]):
        rows.append({"group": grp, "written_standard": ws, "utts": len(g),
                     "wer": jiwer.wer(list(g["ref"]), list(g["hyp"])), "cer": jiwer.cer(list(g["ref"]), list(g["hyp"]))})
    return pd.DataFrame(rows)

def main():
    cfg = load_config()
    s5 = cfg["step5"]
    out = cfg["out_dir"] / "step5"
    east = cfg["analysis"]["east_groups"]
    act = Activations(cfg["out_dir"] / "activations")
    lags = json.loads((out / "selected_lags.json").read_text())

    err = error_rates(cfg)
    err.to_csv(out / "error_rates.csv", index=False)
    wide = err.pivot(index="group", columns="written_standard", values=["wer", "cer"])
    print("greedy error rates per group (all speakers, no LM):")
    print(wide.round(3).to_string())
    bm = err[err["written_standard"] == "bokmål"].set_index("group")

    half = pd.read_csv(out / "split.csv").set_index("speaker")["half"]
    w = act.words.assign(half=act.words["speaker"].map(half))
    meta = w[(w["half"] == "analysis") & ~w["group"].isin(cfg["analysis"]["exclude_groups"])]
    rows = meta.index.to_numpy()
    meta = meta.reset_index(drop=True)

    dist, corr = [], []
    for li, name in enumerate(act.layers):
        for variant in ("lag", "zero", "ctc"):
            x = act.vectors(rows, li, variant, lags[name])
            ref = PairStats(x, meta, cfg["analysis"]["min_dialect_groups_per_word"]).to_reference(meta, east)
            d = 1 - ref.groupby("group")["sim"].mean()
            dist.append(pd.DataFrame({"layer": name, "layer_idx": li, "variant": variant, "group": d.index, "distance": d.values}))
            for metric in ("wer", "cer"):
                a, b = d.to_numpy(), bm.loc[d.index, metric].to_numpy()
                rho = spearmanr(a, b).statistic
                r = pearsonr(a, b).statistic
                corr.append({"layer": name, "layer_idx": li, "variant": variant, "metric": metric, "spearman": rho,
                             "p_spearman": perm_p(a, b, lambda u, v: spearmanr(u, v).statistic, 5000, s5["seed"]),
                             "pearson": r, "p_pearson": perm_p(a, b, lambda u, v: pearsonr(u, v).statistic, 5000, s5["seed"])})
        c = [k for k in corr if k["layer"] == name and k["variant"] == "lag" and k["metric"] == "wer"][0]
        print(f"{name:9s} WER vs distance from East: spearman {c['spearman']:+.2f} (p {c['p_spearman']:.3f}), "
              f"pearson {c['pearson']:+.2f} (p {c['p_pearson']:.3f})", flush=True)
    pd.concat(dist).to_csv(out / "distance_from_east.csv", index=False)
    pd.DataFrame(corr).to_csv(out / "wer_distance_correlation.csv", index=False)

if __name__ == "__main__":
    main()
