"""Logit lens over the reader: at which block do kept and deleted function words diverge?"""
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
outs = {}
hooks = [b.register_forward_hook(lambda m, i, o, k=k: outs.__setitem__(k, o[0].float())) for k, b in enumerate(model.blocks)]
hooks.append(model.proj.register_forward_hook(lambda m, i, o: outs.__setitem__("in", o[0].float())))
head = lambda h: (model.head_proj(h) @ model.unit_emb.t() + model.head_bias).float().log_softmax(-1)
words = set(sys.argv[1:]) or {"så", "og", "jeg"}
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
        lens = {k: head(outs[k][:n]) for k in range(len(model.blocks))}
        for wi, x in enumerate(ref):
            if x in words and spans[wi] and status[wi] in ("delete", "equal"):
                a0, a1 = max(0, spans[wi][0] - 2), min(n, spans[wi][1] + 2)
                row = {"word": x, "status": status[wi], "area": r.area}
                for k, l in lens.items():
                    row[f"b{k}"] = float(1 - l[a0:a1, 0].exp().min())  # max non-blank mass in the window
                rows.append(row)
for h in hooks:
    h.remove()
d = pd.DataFrame(rows)
d.groupby(["word", "status"])[[f"b{k}" for k in range(len(model.blocks))]].median().reset_index().to_csv(ROOT / "results" / "reader_lens.csv", index=False)
print("max non-blank probability in the word window when the head is applied after each reader block:")
print(d.groupby(["word", "status"])[[f"b{k}" for k in range(len(model.blocks))]].median().round(2).to_string())
print(d.groupby(["word", "status"]).size().to_string())
