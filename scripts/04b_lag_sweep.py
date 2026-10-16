import json

import numpy as np
import pandas as pd

from dialect_probing.config import load_config

def pair_sums(x: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    total = x.sum(0)
    within, n_within = 0.0, 0.0
    for g in np.unique(labels):
        s = x[labels == g].sum(0)
        k = (labels == g).sum()
        within += s @ s - (x[labels == g] ** 2).sum()
        n_within += k * (k - 1)
    n = len(x)
    across = total @ total - (x ** 2).sum() - within
    return across / max(n * (n - 1) - n_within, 1), within / max(n_within, 1)

def similarity(x: np.ndarray, meta: pd.DataFrame) -> dict:
    x = x - x.mean(0)
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    cross, within = [], []
    for _, idx in meta.groupby("word").indices.items():
        c, w = pair_sums(x[idx], meta["group"].to_numpy()[idx])
        cross.append(c)
        within.append(w)
    words = meta["word"].to_numpy()
    total = x.sum(0)
    same = sum(x[i].sum(0) @ x[i].sum(0) for i in meta.groupby("word").indices.values())
    n_same = sum(len(i) ** 2 for i in meta.groupby("word").indices.values())
    diff_word = (total @ total - same) / (len(x) ** 2 - n_same)
    return {"same_word_cross_dialect": float(np.mean(cross)), "same_word_within_dialect": float(np.mean(within)),
            "different_word": float(diff_word), "n_types": int(len(np.unique(words)))}

def main():
    cfg = load_config()
    act = cfg["out_dir"] / "activations"
    meta = json.loads((act / "meta.json").read_text())
    words = pd.read_parquet(act / "words.parquet")
    sweep = np.load(act / "lag_sweep.npy", mmap_mode="r")
    keep = ~words["group"].isin(cfg["analysis"]["exclude_groups"])
    n_groups = words.loc[keep, "group"].nunique()
    types = words[keep].groupby("word")["group"].nunique()
    sel = keep & words["word"].isin(types[types == n_groups].index)
    idx = np.flatnonzero(sel)
    sub = words[sel].reset_index(drop=True)
    print(f"{len(types[types == n_groups])} word types found in all {n_groups} dialect groups, {len(idx)} tokens")

    rows = []
    for li, layer in enumerate(meta["sweep_layers"]):
        for si, shift in enumerate(meta["sweep_shifts_frames"]):
            r = similarity(np.asarray(sweep[idx, li, si], dtype=np.float32), sub)
            rows.append({"layer": layer, "shift_s": shift * meta["frame_seconds"], **r})
    df = pd.DataFrame(rows)
    df["cross_minus_different"] = df["same_word_cross_dialect"] - df["different_word"]
    df["within_minus_cross"] = df["same_word_within_dialect"] - df["same_word_cross_dialect"]
    df.to_csv(act / "lag_sweep.csv", index=False)
    print(df.round(4).to_string(index=False))
    print("\npeak shift of same-word cross-dialect similarity (minus different-word baseline):")
    for layer, g in df.groupby("layer"):
        best = g.loc[g["cross_minus_different"].idxmax()]
        print(f"  layer {layer:2d}: +{best['shift_s']:.2f} s ({best['cross_minus_different']:.4f} vs {g.iloc[0]['cross_minus_different']:.4f} at 0 s)")

if __name__ == "__main__":
    main()
