import time

import pandas as pd
import torch
from norsk_asr.evaluate import greedy
from norsk_asr.lm import ctc_labels, make_decoder
from norsk_asr.train import load_checkpoint
from norsk_asr.transcribe import load_audio

from dialect_probing.align import log_probs
from dialect_probing.config import load_config

def main():
    cfg = load_config()
    root = cfg["data_dir"] / "npsc"
    out = cfg["out_dir"] / "npsc"
    out.mkdir(parents=True, exist_ok=True)
    sent = pd.read_parquet(root / "sentences.parquet")
    device = torch.device("cuda")
    model, units = load_checkpoint(cfg["checkpoint"], device)
    decoder = make_decoder(str(cfg["norsk_asr_dir"] / "data" / "lm" / "no4"), ctc_labels(units), 0.6, 1.0)
    t0, rows, failed = time.time(), [], 0
    for i, r in enumerate(sent.itertuples()):
        lp = log_probs(model, load_audio(r.audio), device)
        g = units.decode(greedy(torch.from_numpy(lp)[None], torch.tensor([len(lp)]))[0])
        try:
            lm = decoder.decode(lp, beam_width=64)
        except ValueError:
            lm, failed = g, failed + 1
        rows.append({"sentence_id": r.sentence_id, "data_split": r.data_split, "greedy": g, "lm": lm, "frames": len(lp)})
        if i % 2000 == 0:
            print(f"{i}/{len(sent)} {time.time() - t0:.0f} s", flush=True)
    pd.DataFrame(rows).to_parquet(out / "hyp_norsk-asr.parquet")
    print(f"{len(sent)} sentences in {time.time() - t0:.0f} s; LM decode failed on {failed} (greedy used)")

if __name__ == "__main__":
    main()
