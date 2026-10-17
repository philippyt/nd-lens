import io

import jiwer
import numpy as np
import pandas as pd
from norsk_asr.text import normalize

from dialect_probing.align import frame_window
from dialect_probing.config import load_config
from dialect_probing.spont_encoder import convergence

def correctness(units: pd.DataFrame) -> dict:
    flags = {}
    for u in units.itertuples():
        ref, hyp = u.ref, normalize(u.hyp or "") or "<empty>"
        ok = np.zeros(len(ref.split()), dtype=bool)
        for c in jiwer.process_words(ref, hyp).alignments[0]:
            if c.type == "equal":
                ok[c.ref_start_idx:c.ref_end_idx] = True
        for i, v in enumerate(ok):
            flags[(u.unit, i)] = bool(v)
    return flags

def main():
    cfg = load_config()
    out = cfg["out_dir"] / "spont"
    east = cfg["analysis"]["east_groups"]
    mg = cfg["analysis"]["min_dialect_groups_per_word"]
    utt = pd.read_parquet(cfg["out_dir"] / "align" / "utterances.parquet").rename(columns={"utt_id": "unit"})
    seg = pd.read_parquet(cfg["data_dir"] / "spont" / "segments.parquet")[["seg_id", "ref"]].merge(
        pd.read_parquet(out / "hyp_norsk-asr.parquet").query("model == 'norsk-asr greedy'")[["seg_id", "hyp"]], on="seg_id").rename(columns={"seg_id": "unit"})
    flags = {"read": correctness(utt), "spontaneous": correctness(seg)}
    sets = {}
    for name, p in (("read", out / "enc_read.npz"), ("spontaneous", out / "enc_spont.npz")):
        z = np.load(p, allow_pickle=True)
        m = pd.read_json(io.StringIO(str(z["meta"])))
        m["correct"] = [flags[name].get((u, p_), False) for u, p_ in zip(m["unit"], m["pos"])]
        sets[name] = (z["x"], m)

    print("1. reliability of CTC word windows")
    for name, (x, m) in sets.items():
        print(f"  {name}: {len(m)} words, {m['correct'].mean():.0%} recognised correctly")
        print(f"    alignment score (mean log-prob of the forced path in the window): correct {m.loc[m['correct'], 'align_score'].mean():.2f}, "
              f"misrecognised {m.loc[~m['correct'], 'align_score'].mean():.2f}")
        print(f"    window length (frames): correct {(m['c1'] - m['c0'])[m['correct']].median():.0f}, misrecognised {(m['c1'] - m['c0'])[~m['correct']].median():.0f}")
        if name == "read":
            cov = []
            for r in m.itertuples():
                j0, j1 = frame_window(r.ref_start, r.ref_end, 10 ** 6)
                inter = max(0, min(j1, r.c1) - max(j0, r.c0))
                cov.append(inter / max(j1 - j0, 1))
            m["coverage"] = cov
            print(f"    share of the manual word window covered by the CTC window: correct {m.loc[m['correct'], 'coverage'].mean():.2f}, "
                  f"misrecognised {m.loc[~m['correct'], 'coverage'].mean():.2f}; windows with no overlap: correct "
                  f"{(m.loc[m['correct'], 'coverage'] == 0).mean():.1%}, misrecognised {(m.loc[~m['correct'], 'coverage'] == 0).mean():.1%}")
        for area_name, groups in (("West", ["g06", "g07", "g08", "g09"]), ("East", east)):
            sub = m[m["group"].isin(groups)]
            print(f"    {area_name}: {sub['correct'].mean():.0%} correct, alignment score {sub['align_score'].mean():.2f}")

    rows = []
    for name, (x, m) in sets.items():
        for subset, mask in (("correct", m["correct"].to_numpy()), ("misrecognised", ~m["correct"].to_numpy())):
            for r in convergence(x[mask], m[mask].reset_index(drop=True), mg, 50, east):
                rows.append({"set": name, "subset": subset, **r})
    res = pd.DataFrame(rows)
    res.to_csv(out / "encoder_by_correctness.csv", index=False)
    print("\n2. convergence by recognition outcome: across/within ratio (perm p) | West's extra distance from East")
    for l in (3, 5, 8, 10, 12):
        g = res[res["layer_idx"] == l].set_index(["set", "subset"])
        parts = [f"{s[:5]} {sub[:5]} {g.loc[(s, sub), 'ratio']:.3f} (p {g.loc[(s, sub), 'p_perm']:.2f}) | "
                 f"{g.loc[(s, sub), 'dist_east_west'] - g.loc[(s, sub), 'dist_east_east']:+.3f}"
                 for s in ("read", "spontaneous") for sub in ("correct", "misrecognised")]
        print(f"  L{l:<2d} " + "   ".join(parts))
    print("  word types:", res.groupby(["set", "subset"])["types"].first().to_dict())

if __name__ == "__main__":
    main()
