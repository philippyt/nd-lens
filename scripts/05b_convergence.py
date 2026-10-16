import json

import numpy as np
import pandas as pd

from dialect_probing.config import load_config
from dialect_probing.step5 import Activations, PairStats, summarize, word_categories

VARIANTS = ["lag", "zero", "ctc"]

def analysis_rows(act: Activations, cfg: dict, out) -> pd.DataFrame:
    half = pd.read_csv(out / "split.csv").set_index("speaker")["half"]
    w = act.words.assign(half=act.words["speaker"].map(half))
    return w[(w["half"] == "analysis") & ~w["group"].isin(cfg["analysis"]["exclude_groups"])]

def permutation_gaps(stats: PairStats, meta: pd.DataFrame, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    groups = meta.drop_duplicates("speaker").set_index("speaker")["group"].reindex(stats.speakers).to_numpy()
    gaps = []
    for _ in range(n):
        gaps.append(summarize(stats.per_word(speaker_groups=rng.permutation(groups)))["gap"])
    return np.array(gaps)

def main():
    cfg = load_config()
    s5 = cfg["step5"]
    out = cfg["out_dir"] / "step5"
    act = Activations(cfg["out_dir"] / "activations")
    lags = json.loads((out / "selected_lags.json").read_text())
    spk = pd.read_parquet(cfg["data_dir"] / "nbtale" / "speakers.parquet").set_index("speaker")
    meta = analysis_rows(act, cfg, out)
    rows = meta.index.to_numpy()
    meta = meta.reset_index(drop=True)
    min_groups = cfg["analysis"]["min_dialect_groups_per_word"]
    ntnu = meta["speaker"].map(spk["recording_place"]).eq("NTNU").to_numpy()
    set_a = meta["sentence_set"].eq("a").to_numpy()
    print(f"analysis half: {meta['speaker'].nunique()} speakers, {len(meta)} tokens, {ntnu.mean():.0%} recorded at NTNU")

    freq = act.words["word"].value_counts()
    cats = word_categories(pd.Series(freq.index, index=freq.index), cfg["data_dir"] / "ordbank")
    summary, per_word_all = [], []
    for li, name in enumerate(act.layers):
        for variant in VARIANTS:
            x = act.vectors(rows, li, variant, lags[name])
            stats = PairStats(x, meta, min_groups)
            pw = stats.per_word(meta=meta)
            r = {"layer": name, "layer_idx": li, "variant": variant, "subset": "all", **summarize(pw)}
            if variant == "lag":
                null = permutation_gaps(stats, meta, s5["permutations"], s5["seed"])
                r.update(null_gap_mean=null.mean(), null_gap_sd=null.std(), p_perm=(1 + (null >= r["gap"]).sum()) / (1 + len(null)))
                per_word_all.append(pw.assign(layer=name, layer_idx=li))
                sub = meta[ntnu].reset_index(drop=True)
                summary.append({"layer": name, "layer_idx": li, "variant": variant, "subset": "ntnu",
                                **summarize(PairStats(x[ntnu], sub, min_groups).per_word(meta=sub))})
                sub = meta[set_a].reset_index(drop=True)
                summary.append({"layer": name, "layer_idx": li, "variant": variant, "subset": "set a",
                                **summarize(PairStats(x[set_a], sub, min_groups, exclude_same_sentence=False).per_word(meta=sub))})
            summary.append(r)
        main = [s for s in summary if s["layer"] == name and s["variant"] == "lag" and s["subset"] == "all"][0]
        print(f"{name:9s} within {main['within']:.3f} cross {main['cross']:.3f} gap {main['gap']:.4f} "
              f"ratio {main['ratio']:.3f} null gap {main['null_gap_mean']:+.4f}±{main['null_gap_sd']:.4f} "
              f"p {main['p_perm']:.3f} types {main['types']}", flush=True)

    summary = pd.DataFrame(summary)
    summary.to_csv(out / "convergence.csv", index=False)
    pw = pd.concat(per_word_all, ignore_index=True)
    pw["freq"] = pw["word"].map(freq)
    pw["chars"] = pw["word"].str.replace(" ", "").str.len()
    pw["category"] = pw["word"].map(cats)
    types = pw.drop_duplicates("word")
    edges = types["freq"].quantile([1 / 3, 2 / 3]).to_numpy()
    pw["freq_bin"] = pd.cut(pw["freq"], [0, edges[0], edges[1], np.inf], labels=["rare", "mid", "frequent"])
    pw["length_bin"] = pd.cut(pw["chars"], [0, 3, 6, np.inf], labels=["1 to 3", "4 to 6", "7 plus"])
    pw.to_csv(out / "convergence_per_word.csv", index=False)
    print(f"\nfrequency bin edges (tokens): {edges.round(1).tolist()}")

    strata = []
    for col in ("freq_bin", "length_bin", "category"):
        for (name, li, level), g in pw.groupby(["layer", "layer_idx", col], observed=True):
            strata.append({"stratum": col, "level": level, "layer": name, "layer_idx": li, **summarize(g)})
    strata = pd.DataFrame(strata).sort_values(["stratum", "level", "layer_idx"])
    strata.to_csv(out / "convergence_strata.csv", index=False)
    print("\ntypes per stratum:", strata.query("layer == 'L6'").set_index(["stratum", "level"])["types"].to_dict())
    for col in ("freq_bin", "length_bin", "category"):
        print(f"\nratio (cross / within) by {col}:")
        print(strata[strata["stratum"] == col].pivot(index="layer_idx", columns="level", values="ratio").round(3).to_string())
    print("\nexamples per category:", {c: list(types[types["category"] == c].nlargest(8, "freq")["word"]) for c in types["category"].unique()})
    print("\nratio by variant and subset:")
    print(summary.pivot_table(index="layer_idx", columns=["variant", "subset"], values="ratio").round(3).to_string())

if __name__ == "__main__":
    main()
