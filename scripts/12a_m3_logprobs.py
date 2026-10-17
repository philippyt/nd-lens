import numpy as np
import pandas as pd
import torch
from norsk_asr.train import load_checkpoint
from norsk_asr.transcribe import load_audio

from dialect_probing.align import log_probs
from dialect_probing.config import load_config
from dialect_probing.m3_decode import greedy_words, split_speakers

SR = 16000

def main():
    cfg = load_config()
    out = cfg["out_dir"] / "spont"
    seg = pd.read_parquet(cfg["data_dir"] / "spont" / "segments.parquet")
    seg = seg[seg["set"] == "nbtale_m3"].sort_values(["path", "start"]).reset_index(drop=True)
    split = split_speakers(seg)
    split.to_csv(out / "m3_split.csv", index=False)
    print(split.groupby(["area", "half"]).size().unstack())
    device = torch.device("cuda")
    model, units = load_checkpoint(cfg["checkpoint"], device)
    lps, audio, path = {}, None, None
    for i, r in seg.iterrows():
        if r["path"] != path:
            path, audio = r["path"], load_audio(r["path"])
        a = audio[int(r["start"] * SR):int(np.ceil(r["end"] * SR))]
        lps[f"s{i}"] = log_probs(model, a, device).astype(np.float16)
    np.savez(out / "m3_logprobs.npz", **lps)
    seg[["seg_id", "speaker", "area", "ref"]].to_parquet(out / "m3_logprobs_index.parquet")
    old = pd.read_parquet(out / "hyp_norsk-asr.parquet").query("model == 'norsk-asr greedy'").set_index("seg_id")["hyp"]
    same = [" ".join(w for w, _ in greedy_words(lps[f"s{i}"].astype(np.float32), units.pieces)) == " ".join((old.get(s) or "").split())
            for i, s in enumerate(seg["seg_id"])]
    print(f"{len(seg)} segments, {units.pieces[:3]} ... {len(units)} units; greedy identical to 08b for {np.mean(same):.1%}")

if __name__ == "__main__":
    main()
