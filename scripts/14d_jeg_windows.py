import collections

import jiwer
import numpy as np
import pandas as pd
import torch
from norsk_asr.text import normalize

from dialect_probing.align import ctc_viterbi, word_spans
from dialect_probing.config import ROOT, load_config
from dialect_probing.weight_space import Audio, decode, load_model, module3

def pairs(seg: pd.DataFrame, hyps: list[str]) -> pd.DataFrame:
    c, n = {a: collections.Counter() for a in ("west", "east")}, {"west": 0, "east": 0}
    for r, h in zip(seg.itertuples(), hyps):
        h = " ".join(normalize(h or "").split()) or "<empty>"
        ref, hw = r.ref.split(), h.split()
        n[r.area] += len(ref)
        for a in jiwer.process_words(r.ref, h).alignments[0]:
            rs, hs = ref[a.ref_start_idx:a.ref_end_idx], hw[a.hyp_start_idx:a.hyp_end_idx]
            if a.type == "substitute":
                c[r.area].update(zip(rs, hs))
            elif a.type == "delete":
                c[r.area].update((w, "<del>") for w in rs)
            elif a.type == "insert":
                c[r.area].update(("<ins>", w) for w in hs)
    d = pd.DataFrame([{"ref": k[0], "hyp": k[1], "west_per_1k": 1000 * c["west"][k] / n["west"], "east_per_1k": 1000 * c["east"][k] / n["east"],
                       "west_n": c["west"][k]} for k in set(c["west"]) | set(c["east"])])
    return d.assign(excess=d["west_per_1k"] - d["east_per_1k"]).sort_values("excess", ascending=False)

def main():
    cfg = load_config()
    device = torch.device("cuda")
    model, units, _, _ = load_model(cfg, device)
    seg = module3(cfg, "tune")
    hyps, lps = decode(model, units, seg, Audio(cfg), device, return_lp=True)
    out = cfg["out_dir"] / "weight_space"
    out.mkdir(parents=True, exist_ok=True)
    seg.assign(hyp=[" ".join(normalize(h or "").split()) for h in hyps]).drop(columns=["path"]).to_parquet(out / "tune_hyps.parquet")
    keep = seg["area"].isin(["west", "east"]).to_numpy()
    seg = seg[keep].reset_index(drop=True)
    hyps, lps = [h for h, k in zip(hyps, keep) if k], [x for x, k in zip(lps, keep) if k]

    wp = pairs(seg, hyps)
    wp.head(100).to_csv(ROOT / "results" / "word_pair_excess.csv", index=False)
    print(f"top 40 word pairs explain {wp.head(40)['excess'].sum():.1f} of {wp['excess'].sum():.1f} excess errors per 1000 words")
    print(wp.head(15).round(2).to_string(index=False))

    j, rows, frames = units.pieces.index("▁j"), [], []
    for r, h, lp in zip(seg.itertuples(), hyps, lps):
        lp = lp.astype(np.float32)
        h = " ".join(normalize(h or "").split()) or "<empty>"
        ref = r.ref.split()
        status = ["?"] * len(ref)
        for a in jiwer.process_words(r.ref, h).alignments[0]:
            status[a.ref_start_idx:a.ref_end_idx] = [a.type] * (a.ref_end_idx - a.ref_start_idx)
        targets, tw = [], []
        for wi, w in enumerate(ref):
            ids = units.encode(w)
            targets += ids
            tw += [wi] * len(ids)
        res = ctc_viterbi(lp, targets)
        if res is None:
            continue
        spans = word_spans(res[0], np.array(tw), len(ref))
        near = np.zeros(len(lp), bool)
        for wi, w in enumerate(ref):
            if spans[wi] and w.startswith("j"):
                near[max(0, spans[wi][0] - 3):spans[wi][1] + 3] = True
        frames.append(pd.DataFrame({"area": r.area, "margin": lp[~near, j] - lp[~near, 0]}))
        for wi, w in enumerate(ref):
            if w == "jeg" and spans[wi]:
                win = lp[max(0, spans[wi][0] - 2):spans[wi][1] + 2]
                best = win[:, 1:].max(0)
                rows.append({"area": r.area, "status": status[wi], "max_margin_j": float((win[:, j] - win[:, 0]).max()),
                             "max_p_nonblank": float(np.exp(best.max())), "top_nonblank": units.pieces[int(best.argmax()) + 1]})
    d, f = pd.DataFrame(rows), pd.concat(frames)
    d.to_csv(ROOT / "results" / "jeg_windows.csv", index=False)
    print("\njeg tokens:", d.groupby(["area", "status"]).size().unstack(fill_value=0).to_dict())
    dw = d.query("area == 'west' and status == 'delete'")
    print(f"deleted West jeg: median max p(non-blank) {dw['max_p_nonblank'].median():.2f}, "
          f"top non-blank piece {collections.Counter(dw['top_nonblank']).most_common(5)}")
    for t in (0, 2, 4):
        print(f"bias ▁j +{t}: recovers {(dw['max_margin_j'] > -t).mean():.2f} of deleted West jeg, "
              f"fires on {(f['margin'] > -t).mean():.4f} of other frames ({int((f['margin'] > -t).sum())} frames)")

if __name__ == "__main__":
    main()
