"""Channel-level attribution: in block 5 conv (768 channels into pw2) and block 4 ff1 (3072 units into its second
linear), which units push toward blank more at deleted than at kept så/og, and how much do they push elsewhere?"""
import importlib
import json
import sys

import numpy as np
import pandas as pd
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import jiwer
from dialect_probing.align import ctc_viterbi, word_spans
from dialect_probing.config import ROOT, load_config
from dialect_probing.weight_space import Audio, module3
from norsk_asr.text import normalize

loop = importlib.import_module("15_edit_loop")
cfg = load_config(); dev = torch.device("cuda")
state = json.loads((ROOT / "edits" / "state.json").read_text())
model, units = loop.build(cfg, dev, loop.stack(state))
audio = Audio(cfg); seg = module3(cfg, "tune")
E, Wh = model.unit_emb.detach().float(), model.head_proj.weight.detach().float()
g = (E[0] - E[1:].mean(0)) @ Wh  # blank-vs-rest readout of the final hidden state
SITES = {"b5_conv": (model.blocks[5].conv.pw2, 1.0), "b4_ff1": (model.blocks[4].ff1.net[4], 0.5),
         "b5_ff1": (model.blocks[5].ff1.net[4], 0.5), "b4_ff2": (model.blocks[4].ff2.net[4], 0.5)}
cap = {}
hooks = [lin.register_forward_hook(lambda m, i, o, k=k: cap.__setitem__(k, i[0][0].float())) for k, (lin, _) in SITES.items()]
# per-channel blank push = activation * (W[:, c] . g) * mult (ignores the final LayerNorm scale; same for all channels)
readout = {k: (lin.weight.detach().float().T @ g) * mult for k, (lin, mult) in SITES.items()}
acc = {k: {"delete": 0, "equal": 0, "all": 0} for k in SITES}
cnt = {"delete": 0, "equal": 0, "all": 0}
words = set(sys.argv[1].split(",")) if len(sys.argv) > 1 else {"så", "og"}
TAG = sys.argv[2] if len(sys.argv) > 2 else "channels"
with torch.no_grad():
    for r in seg.itertuples():
        w = torch.from_numpy(audio(r.seg_id))[None].to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lp, ol = model(w, torch.tensor([w.shape[1]], device=dev))
        n = int(ol[0]); lp = lp[0, :n].float()
        for k in SITES:
            acc[k]["all"] = acc[k]["all"] + (cap[k][:n] * readout[k]).sum(0)
        cnt["all"] += n
        ref = r.ref.split()
        if not words & set(ref):
            continue
        hyp = " ".join(normalize(units.decode(loop.greedy_ids(lp))).split()) or "<empty>"
        status = ["?"] * len(ref)
        for a in jiwer.process_words(r.ref, hyp).alignments[0]:
            status[a.ref_start_idx:a.ref_end_idx] = [a.type] * (a.ref_end_idx - a.ref_start_idx)
        tg, tw = [], []
        for wi, x in enumerate(ref):
            ids = units.encode(x); tg += ids; tw += [wi] * len(ids)
        res = ctc_viterbi(lp.cpu().numpy(), tg)
        if res is None:
            continue
        spans = word_spans(res[0], np.array(tw), len(ref))
        for wi, x in enumerate(ref):
            if x in words and spans[wi] and status[wi] in ("delete", "equal"):
                a0, a1 = spans[wi][0], min(n, spans[wi][1])
                for k in SITES:
                    acc[k][status[wi]] = acc[k][status[wi]] + (cap[k][a0:a1] * readout[k]).mean(0)
                cnt[status[wi]] += 1
for h in hooks:
    h.remove()
out = {}
for k in SITES:
    dl, eq, al = acc[k]["delete"] / cnt["delete"], acc[k]["equal"] / cnt["equal"], acc[k]["all"] / cnt["all"]
    diff = (dl - eq).cpu()
    order = torch.argsort(diff, descending=True)
    print(f"\n{k}: {len(diff)} units; total deleted-minus-kept blank push {diff.sum():+.2f}, positive part {diff.clamp(min=0).sum():.2f}")
    top = order[:12]
    print("  top units: diff / push on kept / push on all frames")
    for c in top:
        print(f"   {int(c):5d} {diff[c]:+.3f} {eq[c].item():+.3f} {al[c].item():+.3f}")
    cum = diff[order].cumsum(0)
    print("  cumulative diff of top 4/8/16/32/64:", [round(cum[i - 1].item(), 2) for i in (4, 8, 16, 32, 64)])
    out[k] = {"diff": diff, "kept": eq.cpu(), "all": al.cpu()}
torch.save(out, ROOT / "outputs" / "edit_loop" / f"{TAG}.pt")
sel = [{"site": k, "unit": int(u), "diff": float(v["diff"][u]), "kept": float(v["kept"][u]), "all": float(v["all"][u])}
       for k, v in out.items() for u in torch.nonzero((v["diff"] > 0.08) & (v["kept"].abs() < 0.05) & (v["all"].abs() < 0.02)).flatten()]
pd.DataFrame(sel).to_csv(ROOT / "results" / f"suppression_units_{TAG}.csv", index=False)
print(pd.DataFrame(sel).round(3).to_string(index=False))
print("counts", cnt)
