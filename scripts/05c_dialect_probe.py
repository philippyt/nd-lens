import json

import numpy as np
import pandas as pd
import torch

from dialect_probing.config import load_config
from dialect_probing.step5 import Activations, probe_accuracy, speaker_folds

def cv_accuracy(x, y, folds, n_classes, device) -> float:
    return float(np.mean([probe_accuracy(x[folds != f], y[folds != f], x[folds == f], y[folds == f], n_classes, device)
                          for f in np.unique(folds)]))

def main():
    cfg = load_config()
    s5 = cfg["step5"]
    out = cfg["out_dir"] / "step5"
    act = Activations(cfg["out_dir"] / "activations")
    lags = json.loads((out / "selected_lags.json").read_text())
    spk = pd.read_parquet(cfg["data_dir"] / "nbtale" / "speakers.parquet")
    half = pd.read_csv(out / "split.csv").set_index("speaker")["half"]
    w = act.words.assign(half=act.words["speaker"].map(half))
    meta = w[(w["half"] == "analysis") & ~w["group"].isin(cfg["analysis"]["exclude_groups"])]
    rows = meta.index.to_numpy()
    groups = sorted(meta["group"].unique())
    y = pd.Categorical(meta["group"], categories=groups).codes
    folds = meta["speaker"].map(speaker_folds(spk[spk["speaker"].isin(meta["speaker"])], s5["probe_folds"], s5["seed"])).to_numpy()
    ntnu = meta["speaker"].map(spk.set_index("speaker")["recording_place"]).eq("NTNU").to_numpy()
    chance = pd.Series(y).value_counts(normalize=True).max()
    print(f"{len(groups)} dialect groups, {len(meta)} tokens, {meta['speaker'].nunique()} speakers, "
          f"{s5['probe_folds']} speaker folds, majority baseline {chance:.3f}")

    device = torch.device("cuda")
    res = []
    for li, name in enumerate(act.layers):
        r = {"layer": name, "layer_idx": li}
        for variant in ("lag", "zero", "ctc"):
            x = act.vectors(rows, li, variant, lags[name])
            r[variant] = cv_accuracy(x, y, folds, len(groups), device)
            if variant == "lag":
                r["lag_ntnu"] = cv_accuracy(x[ntnu], y[ntnu], folds[ntnu], len(groups), device)
        res.append(r)
        print(f"{name:9s} " + "  ".join(f"{k} {v:.3f}" for k, v in r.items() if k not in ("layer", "layer_idx")), flush=True)
    df = pd.DataFrame(res)
    df["chance"] = chance
    df["chance_ntnu"] = pd.Series(y[ntnu]).value_counts(normalize=True).max()
    df.to_csv(out / "dialect_probe.csv", index=False)

if __name__ == "__main__":
    main()
