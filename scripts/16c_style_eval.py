import collections
import importlib.util

import jiwer
import numpy as np
import pandas as pd
import torch
from norsk_asr.text import normalize
from norsk_asr.train import load_checkpoint

from dialect_probing.align import ctc_viterbi, word_spans
from dialect_probing.config import ROOT, load_config
from dialect_probing.weight_space import Audio, decode, gap_ci, module3, score, summary

SPOKEN = {"så", "ja", "jo", "da", "du"}
B = 2000

def token_outcomes(seg: pd.DataFrame, hyps: list[str]) -> pd.DataFrame:
    rows = []
    for r, h in zip(seg.itertuples(), hyps):
        h = " ".join(normalize(h or "").split()) or "<empty>"
        ref, hw = r.ref.split(), h.split()
        for a in jiwer.process_words(r.ref, h).alignments[0]:
            for k in range(a.ref_start_idx, a.ref_end_idx):
                hyp = hw[a.hyp_start_idx + k - a.ref_start_idx] if a.type == "substitute" else ""
                rows.append({"seg_id": r.seg_id, "speaker": r.speaker, "area": r.area, "pos": k, "word": ref[k], "op": a.type, "hyp": hyp})
    return pd.DataFrame(rows)

def rate_ci(a: pd.DataFrame, b: pd.DataFrame, hit, mask, seed: int = 0):
    """Pooled rate in a and b over tokens where mask holds, and a speaker-bootstrap CI of b minus a (paired)."""
    sa, sb = a[mask(a)].assign(h=lambda d: hit(d)), b[mask(b)].assign(h=lambda d: hit(d))
    ga, gb = sa.groupby("speaker")["h"].agg(["sum", "size"]), sb.groupby("speaker")["h"].agg(["sum", "size"])
    spk = ga.index.intersection(gb.index).to_numpy()
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(B):
        p = rng.choice(spk, len(spk))
        d.append(gb.loc[p, "sum"].sum() / gb.loc[p, "size"].sum() - ga.loc[p, "sum"].sum() / ga.loc[p, "size"].sum())
    return sa["h"].mean(), sb["h"].mean(), *np.percentile(d, [2.5, 97.5])

@torch.no_grad()
def unit_push(model, units, seg, audio, device) -> dict:
    """Blank push of the 19 edit-04 units at så/og tokens (same unit indices), split by kept/deleted."""
    spec = importlib.util.spec_from_file_location("e04", ROOT / "edits" / "04_flip_suppression_units.py")
    e04 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(e04)
    g = (model.unit_emb[0] - model.unit_emb[1:].mean(0)).float() @ model.head_proj.weight.float()
    cap, hooks, readout = {}, [], {}
    for (blk, name), cols in e04.UNITS.items():
        lin = getattr(model.blocks[blk], name).net[4]
        readout[(blk, name)] = ((lin.weight.float().T @ g) * 0.5)[cols]
        hooks.append(lin.register_forward_hook(lambda m, i, o, k=(blk, name), c=cols: cap.__setitem__(k, i[0][0][:, c].float())))
    acc = collections.defaultdict(list)
    for r in seg[seg["ref"].str.contains(r"\b(?:så|og)\b")].itertuples():
        w = torch.from_numpy(audio(r.seg_id))[None].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lp, ol = model(w, torch.tensor([w.shape[1]], device=device))
        n = int(ol[0])
        lp = lp[0, :n].float()
        ref = r.ref.split()
        ids = torch.unique_consecutive(lp.argmax(-1)).tolist()
        hyp = " ".join(normalize(units.decode([u for u in ids if u])).split()) or "<empty>"
        status = ["?"] * len(ref)
        for a in jiwer.process_words(r.ref, hyp).alignments[0]:
            status[a.ref_start_idx:a.ref_end_idx] = [a.type] * (a.ref_end_idx - a.ref_start_idx)
        tg, tw = [], []
        for wi, x in enumerate(ref):
            t = units.encode(x)
            tg += t
            tw += [wi] * len(t)
        vit = ctc_viterbi(lp.cpu().numpy(), tg)
        if vit is None:
            continue
        spans = word_spans(vit[0], np.array(tw), len(ref))
        for wi, x in enumerate(ref):
            if x in ("så", "og") and spans[wi]:
                a0, a1 = spans[wi]
                push = sum(float((cap[k][a0:a1] * readout[k]).mean(0).sum()) for k in cap)
                acc[status[wi]].append(push)
                acc["all"].append(push)
    for h in hooks:
        h.remove()
    m = {k: float(np.mean(v)) for k, v in acc.items()}
    return {"push_all": m["all"], "push_deleted": m.get("delete", np.nan), "push_kept": m.get("equal", np.nan),
            "deleted_minus_kept": m.get("delete", np.nan) - m.get("equal", np.nan), "deleted_share": len(acc["delete"]) / len(acc["all"])}

def main():
    cfg = load_config()
    device = torch.device("cuda")
    audio = Audio(cfg)
    out = cfg["out_dir"] / "style"
    runs = {"base": cfg["checkpoint"], "edited": out / "edited" / "run", "verbatim": out / "verbatim" / "run"}
    rows, tok, scored, units_rows = [], {}, {}, []
    for name, path in runs.items():
        model, units = load_checkpoint(path, device)
        for half in ("tune", "test"):
            seg = module3(cfg, half)
            hyps = decode(model, units, seg, audio, device)
            sc = score(seg, hyps)
            scored[(name, half)], tok[(name, half)] = sc, token_outcomes(seg, hyps)
            pd.DataFrame({"seg_id": seg["seg_id"], "hyp": hyps}).to_parquet(out / f"hyps_{name}_{half}.parquet")
            s = summary(sc)
            t = tok[(name, half)]
            sp = t[t["word"].isin(SPOKEN)]
            du = t[t["word"] == "du"]
            rows.append({"model": name, "half": half, **s, "spoken_del_rate": (sp["op"] == "delete").mean(),
                         "du_as_man": (du["hyp"] == "man").mean(), "del_rate_all": (t["op"] == "delete").mean()})
            print(f"{name:8s} {half}: WER {100 * s['all']:.1f} West {100 * s['west']:.1f} East {100 * s['east']:.1f} gap {100 * s['gap']:+.1f}  "
                  f"spoken-word deletions {100 * rows[-1]['spoken_del_rate']:.1f}%  du->man {100 * rows[-1]['du_as_man']:.1f}%", flush=True)
        units_rows.append({"model": name, **unit_push(model, units, module3(cfg, "tune"), audio, device)})
        del model
        torch.cuda.empty_cache()
    res = pd.DataFrame(rows)
    res.to_csv(ROOT / "results" / "style_finetune.csv", index=False)
    pd.DataFrame(units_rows).to_csv(ROOT / "results" / "style_units.csv", index=False)

    tests = []
    for half in ("test", "tune"):
        for a, b in (("edited", "verbatim"), ("base", "verbatim"), ("base", "edited")):
            ta, tb = tok[(a, half)], tok[(b, half)]
            p1 = rate_ci(ta, tb, lambda d: d["op"] == "delete", lambda d: d["word"].isin(SPOKEN))
            p2 = rate_ci(ta, tb, lambda d: d["hyp"] == "man", lambda d: d["word"] == "du")
            lo, hi = gap_ci(scored[(a, half)], scored[(b, half)])
            ga, gb = summary(scored[(a, half)])["gap"], summary(scored[(b, half)])["gap"]
            tests.append({"half": half, "from": a, "to": b, "spoken_del_from": p1[0], "spoken_del_to": p1[1], "spoken_del_diff_lo": p1[2],
                          "spoken_del_diff_hi": p1[3], "du_man_from": p2[0], "du_man_to": p2[1], "du_man_diff_lo": p2[2], "du_man_diff_hi": p2[3],
                          "gap_from": ga, "gap_to": gb, "gap_change_lo": lo, "gap_change_hi": hi})
    tests = pd.DataFrame(tests)
    tests.to_csv(ROOT / "results" / "style_tests.csv", index=False)
    pct = lambda x: f"{100 * x:+.1f}"
    print("\npaired comparisons (to minus from), 95% speaker-bootstrap CIs:")
    for r in tests.itertuples():
        print(f"  {r.half} {r._3:>8s} -> {r.to:8s} spoken deletions {100 * r.spoken_del_from:.1f} -> {100 * r.spoken_del_to:.1f} "
              f"[{pct(r.spoken_del_diff_lo)},{pct(r.spoken_del_diff_hi)}]  du->man {100 * r.du_man_from:.1f} -> {100 * r.du_man_to:.1f} "
              f"[{pct(r.du_man_diff_lo)},{pct(r.du_man_diff_hi)}]  gap {pct(r.gap_from)} -> {pct(r.gap_to)} [{pct(r.gap_change_lo)},{pct(r.gap_change_hi)}]")
    print("\nsuppression units at så/og (tune half):")
    print(pd.DataFrame(units_rows).round(3).to_string(index=False))

if __name__ == "__main__":
    main()
