import json

import numpy as np
import pandas as pd
import torch

from dialect_probing.config import load_config
from dialect_probing.step5 import Activations, probe_accuracy, speaker_folds, speaker_split

def main():
    cfg = load_config()
    s5 = cfg["step5"]
    out = cfg["out_dir"] / "step5"
    out.mkdir(parents=True, exist_ok=True)
    act = Activations(cfg["out_dir"] / "activations")
    spk = pd.read_parquet(cfg["data_dir"] / "nbtale" / "speakers.parquet")
    half = speaker_split(spk, s5["seed"])
    half.rename_axis("speaker").reset_index().to_csv(out / "split.csv", index=False)
    print("speakers per half:", half.value_counts().to_dict())

    words = act.words.assign(half=act.words["speaker"].map(half))
    sel = words[words["half"] == "select"]
    counts = sel["word"].value_counts()
    classes = counts[counts >= s5["min_tokens_per_word_class"]].index
    sel = sel[sel["word"].isin(classes)]
    y = pd.Categorical(sel["word"], categories=classes).codes
    folds = sel["speaker"].map(speaker_folds(spk[spk["speaker"].isin(sel["speaker"])], 2, s5["seed"])).to_numpy()
    rows_idx = sel.index.to_numpy()
    print(f"word identity probe: {len(classes)} word classes, {len(sel)} tokens, chance {counts[classes].max() / len(sel):.3f} (majority)")

    device = torch.device("cuda")
    res = []
    for li, name in enumerate(act.layers):
        for shift in act.meta["sweep_shifts_frames"]:
            x = act.vectors(rows_idx, li, "lag", shift)
            acc = np.mean([probe_accuracy(x[folds != f], y[folds != f], x[folds == f], y[folds == f], len(classes), device)
                           for f in (0, 1)])
            res.append({"layer": name, "layer_idx": li, "shift_frames": shift, "shift_s": shift * act.meta["frame_seconds"], "word_acc": acc})
        g = pd.DataFrame(res).query("layer == @name")
        best = g.loc[g["word_acc"].idxmax()]
        print(f"{name:9s} best shift +{best['shift_s']:.2f} s  acc {best['word_acc']:.3f}  (0 s: {g.iloc[0]['word_acc']:.3f})", flush=True)
    df = pd.DataFrame(res)
    df.to_csv(out / "lag_selection.csv", index=False)
    chosen = df.loc[df.groupby("layer_idx")["word_acc"].idxmax()].sort_values("layer_idx")
    (out / "selected_lags.json").write_text(json.dumps({r["layer"]: int(r["shift_frames"]) for _, r in chosen.iterrows()}, indent=1))

if __name__ == "__main__":
    main()
