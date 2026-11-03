"""Which branch of reader blocks 3-5 pushes deleted så/og toward blank? Direct effect of each residual update on
the blank-vs-rest logit, read through the final LayerNorm (frozen at its observed scale)."""
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
# logit(blank) - mean logit(non-blank), as a direction in the final hidden space (input to head_proj)
g = (E[0] - E[1:].mean(0)) @ Wh
cap = {}
BL = [3, 4, 5]
hooks = []
for k in BL:
    b = model.blocks[k]
    for name in ("ff1", "attn", "conv", "ff2"):
        hooks.append(getattr(b, name).register_forward_hook(lambda m, i, o, key=(k, name): cap.__setitem__(key, o[0].float())))
    hooks.append(b.norm.register_forward_hook(lambda m, i, o, k=k: cap.__setitem__((k, "pre_norm"), i[0][0].float())))
words = {"så", "og"}
rows = []
with torch.no_grad():
    for r in seg.itertuples():
        ref = r.ref.split()
        if not words & set(ref):
            continue
        w = torch.from_numpy(audio(r.seg_id))[None].to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lp, ol = model(w, torch.tensor([w.shape[1]], device=dev))
        n = int(ol[0]); lp = lp[0, :n].float()
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
                row = {"word": x, "status": status[wi]}
                for k in BL:
                    pre = cap[(k, "pre_norm")][a0:a1]
                    s = model.blocks[k].norm.weight.float() / pre.std(-1, keepdim=True)  # LayerNorm scale at these frames
                    for name, mult in (("ff1", 0.5), ("attn", 1.0), ("conv", 1.0), ("ff2", 0.5)):
                        u = cap[(k, name)][a0:a1] * mult
                        # only the last block's norm feeds the head directly; earlier blocks pass through later ones
                        row[f"b{k}_{name}"] = float(((u - u.mean(-1, keepdim=True)) * s @ (g if k == 5 else g)).mean())
                rows.append(row)
for h in hooks:
    h.remove()
d = pd.DataFrame(rows)
cols = [c for c in d.columns if c.startswith("b")]
m = d.groupby(["word", "status"])[cols].mean()
m.reset_index().to_csv(ROOT / "results" / "reader_branches.csv", index=False)
print("push toward blank (blank minus mean other logit), mean over word frames; block 5 is a direct effect, 3-4 approximate:")
print(m.T.round(2).to_string())
print("\ndeleted minus kept:")
print((m.xs("delete", level="status") - m.xs("equal", level="status")).T.round(2).to_string())
