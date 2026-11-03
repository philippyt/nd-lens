"""Causal check of a candidate edit at så/og windows: change in blank log-prob and in deletions, vs other frames."""
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
words = set(sys.argv[2].split(",")) if len(sys.argv) > 2 else {"så", "og"}
seg = seg[seg["ref"].map(lambda r: bool(words & set(r.split())))]


def run():
    res, other = [], []
    with torch.no_grad():
        for r in seg.itertuples():
            w = torch.from_numpy(audio(r.seg_id))[None].to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lp, ol = model(w, torch.tensor([w.shape[1]], device=dev))
            n = int(ol[0]); lp = lp[0, :n].float()
            ref = r.ref.split()
            hyp = " ".join(normalize(units.decode(loop.greedy_ids(lp))).split()) or "<empty>"
            status = ["?"] * len(ref)
            for a in jiwer.process_words(r.ref, hyp).alignments[0]:
                status[a.ref_start_idx:a.ref_end_idx] = [a.type] * (a.ref_end_idx - a.ref_start_idx)
            tg, tw = [], []
            for wi, x in enumerate(ref):
                ids = units.encode(x); tg += ids; tw += [wi] * len(ids)
            vit = ctc_viterbi(lp.cpu().numpy(), tg)
            if vit is None:
                continue
            spans = word_spans(vit[0], np.array(tw), len(ref))
            mask = np.zeros(n, bool)
            for wi, x in enumerate(ref):
                if x in words and spans[wi]:
                    a0, a1 = spans[wi]
                    mask[a0:a1] = True
                    res.append({"seg": r.seg_id, "wi": wi, "status": status[wi], "blank": float(lp[a0:a1, 0].mean())})
            other.append(float(lp[torch.from_numpy(~mask).to(dev), 0].mean()))
    return pd.DataFrame(res), np.mean(other)


a, oa = run()
with torch.no_grad():
    loop.load_edit(Path(sys.argv[1]).resolve()).apply(model, units)
b, ob = run()
m = a.merge(b, on=["seg", "wi"], suffixes=("", "_edit"))
print(f"{sorted(words)}: mean log p(blank) at word frames: {m.blank.mean():.3f} -> {m.blank_edit.mean():.3f}; other frames {oa:.3f} -> {ob:.3f}")
print("by original status, change in mean log p(blank):", m.groupby("status").apply(lambda g: round((g.blank_edit - g.blank).mean(), 3)).to_dict())
print("deleted before:", (m.status == "delete").sum(), " after:", (m.status_edit == "delete").sum())
print(pd.crosstab(m.status, m.status_edit).to_string())
ct = pd.crosstab(m.status, m.status_edit).stack().rename("tokens").reset_index()
ct.assign(edit=Path(sys.argv[1]).stem, words=",".join(sorted(words))).to_csv(ROOT / "results" / f"edit_effect_{Path(sys.argv[1]).stem}_{'-'.join(sorted(words))}.csv", index=False)
