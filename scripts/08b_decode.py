import time

import numpy as np
import pandas as pd
import torch
from norsk_asr.evaluate import greedy
from norsk_asr.lm import ctc_labels, make_decoder
from norsk_asr.train import load_checkpoint
from norsk_asr.transcribe import load_audio

from dialect_probing.align import log_probs
from dialect_probing.config import load_config

SR = 16000

def segments_audio(seg: pd.DataFrame):
    cache = {}
    for _, r in seg.iterrows():
        if r["path"] not in cache:
            cache = {r["path"]: load_audio(r["path"])}
        yield r["seg_id"], cache[r["path"]][int(r["start"] * SR):int(np.ceil(r["end"] * SR))]

def main():
    cfg = load_config()
    out = cfg["out_dir"] / "spont"
    seg = pd.read_parquet(cfg["data_dir"] / "spont" / "segments.parquet").sort_values(["path", "start"])
    device = torch.device("cuda")
    model, units = load_checkpoint(cfg["checkpoint"], device)
    decoder = make_decoder(str(cfg["norsk_asr_dir"] / "data" / "lm" / "no4"), ctc_labels(units), 0.6, 1.0)
    t0 = time.time()
    rows = []
    for sid, a in segments_audio(seg):
        lp = log_probs(model, a, device)
        g = units.decode(greedy(torch.from_numpy(lp)[None], torch.tensor([len(lp)]))[0])
        rows.append({"seg_id": sid, "norsk-asr greedy": g, "norsk-asr lm": decoder.decode(lp, beam_width=64)})
    df = pd.DataFrame(rows).melt("seg_id", var_name="model", value_name="hyp")
    df.to_parquet(out / "hyp_norsk-asr.parquet")
    print(f"{len(seg)} segments, {(seg['end'] - seg['start']).sum() / 3600:.2f} h in {time.time() - t0:.0f} s")

if __name__ == "__main__":
    main()
