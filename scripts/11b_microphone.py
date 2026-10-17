import time

import jiwer
import numpy as np
import pandas as pd
import torch
from norsk_asr.evaluate import greedy
from norsk_asr.text import normalize
from norsk_asr.train import load_checkpoint
from norsk_asr.transcribe import load_audio

from dialect_probing.align import log_probs
from dialect_probing.config import load_config

SR = 16000
AREAS = ["east", "west", "central", "north", "south"]
B = 10000

def main():
    cfg = load_config()
    out = cfg["out_dir"] / "spont"
    seg = pd.read_parquet(cfg["data_dir"] / "spont" / "segments.parquet")
    seg = seg[(seg["set"] == "nbtale_m3") & seg["area"].isin(AREAS)].sort_values(["path", "start"])
    shure_root = cfg["data_dir"] / "nbtale" / "shure"
    path = out / "hyp_shure.parquet"
    if not path.exists():
        device = torch.device("cuda")
        model, units = load_checkpoint(cfg["checkpoint"], device)
        rows, cache, t0 = [], {}, time.time()
        for r in seg.itertuples():
            p = str(shure_root / "/".join(r.path.split("/")[-3:]))
            if p not in cache:
                cache = {p: load_audio(p)}
            a = cache[p][int(r.start * SR):int(np.ceil(r.end * SR))]
            lp = log_probs(model, a, device)
            rows.append({"seg_id": r.seg_id, "hyp": units.decode(greedy(torch.from_numpy(lp)[None], torch.tensor([len(lp)]))[0])})
        pd.DataFrame(rows).to_parquet(path)
        print(f"decoded {len(rows)} Shure segments in {time.time() - t0:.0f} s")
    hyp_s = pd.read_parquet(path).rename(columns={"hyp": "shure"})
    hyp_h = pd.read_parquet(out / "hyp_norsk-asr.parquet").query("model == 'norsk-asr greedy'")[["seg_id", "hyp"]].rename(columns={"hyp": "sennheiser"})
    d = seg.merge(hyp_s, on="seg_id").merge(hyp_h, on="seg_id")
    rows = []
    for mic in ("sennheiser", "shure"):
        for r in d.itertuples():
            o = jiwer.process_words(r.ref, normalize(getattr(r, mic) or "") or "<empty>")
            rows.append({"mic": mic, "speaker": r.speaker, "area": r.area, "words": o.hits + o.substitutions + o.deletions,
                         "err": o.substitutions + o.deletions + o.insertions})
    t = pd.DataFrame(rows).groupby(["mic", "area", "speaker"])[["words", "err"]].sum().reset_index()
    rng = np.random.default_rng(0)
    print("WER % by microphone and area (speakers):")
    print((t.groupby(["area", "mic"]).apply(lambda g: 100 * g["err"].sum() / g["words"].sum()).unstack("mic")).round(1).to_string())
    for mic, g in t.groupby("mic"):
        w, e = g[g["area"] == "west"], g[g["area"] == "east"]
        bw = w[["err", "words"]].to_numpy(float)[rng.integers(0, len(w), (B, len(w)))].sum(1)
        be = e[["err", "words"]].to_numpy(float)[rng.integers(0, len(e), (B, len(e)))].sum(1)
        dd = bw[:, 0] / bw[:, 1] - be[:, 0] / be[:, 1]
        v = w["err"].sum() / w["words"].sum() - e["err"].sum() / e["words"].sum()
        print(f"West minus East, {mic:10s}: {100 * v:+.1f} points [{100 * np.percentile(dd, 2.5):+.1f}, {100 * np.percentile(dd, 97.5):+.1f}]")
    t.to_csv(out / "m3_microphone.csv", index=False)

if __name__ == "__main__":
    main()
