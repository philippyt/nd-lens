import time

import jiwer
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from norsk_asr.evaluate import greedy
from norsk_asr.train import load_checkpoint
from norsk_asr.transcribe import load_audio

from dialect_probing.align import log_probs
from dialect_probing.config import load_config
from dialect_probing.nbtale import wav_path

CONDITIONS = {"baseline": {}, "L7-9 halved": {7: 0.5, 8: 0.5, 9: 0.5}, "L7-9 removed": {7: 0, 8: 0, 9: 0},
              "L0-2 removed": {0: 0, 1: 0, 2: 0}}

def mix_logits(base: torch.Tensor, scale: dict) -> torch.Tensor:
    p = torch.softmax(base, 0).clone()
    for i, s in scale.items():
        p[i] *= s
    return torch.log((p / p.sum()).clamp_min(1e-30))

def main():
    cfg = load_config()
    out = cfg["out_dir"] / "step5"
    nb = cfg["data_dir"] / "nbtale"
    half = pd.read_csv(out / "split.csv").set_index("speaker")["half"]
    ref = pd.read_parquet(cfg["out_dir"] / "align" / "utterances.parquet")[["utt_id", "ref"]]
    utt = pd.read_parquet(nb / "utterances.parquet")[["utt_id", "speaker", "group", "written_standard"]].merge(ref, on="utt_id")
    utt = utt[(utt["speaker"].map(half) == "analysis") & ~utt["group"].isin(cfg["analysis"]["exclude_groups"])]
    print(f"analysis half: {len(utt)} utterances, {utt['speaker'].nunique()} speakers")

    device = torch.device("cuda")
    model, units = load_checkpoint(cfg["checkpoint"], device)
    base = model.layer_weights.detach().clone()
    audio = {u: load_audio(str(wav_path(nb / "raw", u))) for u in utt["utt_id"]}
    rows = []
    for cond, scale in CONDITIONS.items():
        with torch.no_grad():
            model.layer_weights.copy_(mix_logits(base, scale))
        w = torch.softmax(model.layer_weights.detach(), 0).cpu().numpy()
        t0 = time.time()
        hyps = []
        for u in utt["utt_id"]:
            lp = log_probs(model, audio[u], device)
            hyps.append(units.decode(greedy(torch.from_numpy(lp)[None], torch.tensor([len(lp)]))[0]))
        u = utt.assign(hyp=hyps)
        for (grp, ws), g in list(u.groupby(["group", "written_standard"])) + [(("all", ws), g) for ws, g in u.groupby("written_standard")]:
            rows.append({"condition": cond, "group": grp, "written_standard": ws, "utts": len(g),
                         "wer": jiwer.wer(list(g["ref"]), list(g["hyp"])), "cer": jiwer.cer(list(g["ref"]), list(g["hyp"]))})
        bm = u[u["written_standard"] == "bokmål"]
        print(f"{cond:13s} weights L0-2 {w[:3].sum():.3f} L5-6 {w[5:7].sum():.3f} L7-9 {w[7:10].sum():.3f} L12 {w[12]:.3f}  "
              f"Bokmål WER {jiwer.wer(list(bm['ref']), list(bm['hyp'])):.4f}  all WER {jiwer.wer(list(u['ref']), list(u['hyp'])):.4f}  "
              f"({time.time() - t0:.0f} s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "mix_ablation.csv", index=False)
    bm = df[df["written_standard"] == "bokmål"].pivot(index="group", columns="condition", values="wer")[list(CONDITIONS)]
    for c in list(CONDITIONS)[1:]:
        bm[f"d {c}"] = bm[c] - bm["baseline"]
    print("\nBokmål WER per region (analysis half):")
    print((bm * 100).round(2).to_string())
    dist = pd.read_csv(out / "distance_from_east.csv").query("variant == 'lag' and layer == 'L8'").set_index("group")["distance"]
    g = bm.drop(index="all")
    for c in list(CONDITIONS)[1:]:
        print(f"WER change under {c} vs L8 distance from East: spearman {spearmanr(dist[g.index], g[f'd {c}']).statistic:+.2f}")

if __name__ == "__main__":
    main()
