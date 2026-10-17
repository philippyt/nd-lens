import numpy as np
import pandas as pd
import torch
from norsk_asr.lm import ctc_labels, make_decoder
from norsk_asr.train import load_checkpoint
from norsk_asr.transcribe import load_audio

from dialect_probing.align import log_probs
from dialect_probing.config import ROOT, load_config
from dialect_probing.m3_decode import B, counts, greedy_words, lm_decode

VARIANTS = {"greedy": None, "lm as shipped (a 0.6, b 1)": (0.6, 1.0, 0.0), "lm retuned (a 0.4, b 2)": (0.4, 2.0, 0.0),
            "lm retuned + blank penalty 1.0": (0.4, 2.0, 1.0)}

def wer(c: pd.DataFrame, col: str = "err") -> float:
    return c[col].sum() / c["words"].sum()

def main():
    cfg = load_config()
    seg = pd.read_parquet(cfg["data_dir"] / "spont" / "segments.parquet")
    seg = seg[(seg["set"] == "nb_samtale") & seg["seg_id"].str.startswith("data/test/bm/")].reset_index(drop=True)
    print(f"NB Samtale Bokmål test split: {len(seg)} segments, {seg['speaker'].nunique()} speakers, "
          f"{(seg['end'] - seg['start']).sum() / 3600:.2f} h; speakers by area {seg.drop_duplicates('speaker')['area'].value_counts().to_dict()}")
    device = torch.device("cuda")
    model, units = load_checkpoint(cfg["checkpoint"], device)
    lps = [log_probs(model, load_audio(p), device) for p in seg["path"]]
    dec = make_decoder(str(cfg["norsk_asr_dir"] / "data" / "lm" / "no4"), ctc_labels(units), 0.6, 1.0)
    idx = list(range(len(lps)))
    res, base = [], None
    rng = np.random.default_rng(0)
    for name, p in VARIANTS.items():
        if p is None:
            words = [greedy_words(lp, units.pieces) for lp in lps]
        else:
            dec.reset_params(alpha=p[0], beta=p[1])
            words = lm_decode(dec, lps, idx, p[2])
        c = counts(seg, words).set_index("speaker")
        if base is None:
            base = c
        a = c[["err", "words"]].join(base[["err", "words"]], rsuffix="_b").to_numpy(float)
        s = a[rng.integers(0, len(a), (B, len(a)))].sum(1)
        lo, hi = np.percentile(s[:, 0] / s[:, 1] - s[:, 2] / s[:, 3], [2.5, 97.5])
        res.append({"variant": name, "wer": wer(c), "change_vs_greedy": wer(c) - wer(base), "lo": lo, "hi": hi,
                    "sub": wer(c, "sub"), "del": wer(c, "del"), "ins": wer(c, "ins"),
                    **{f"wer_{k}": wer(g) for k, g in c.groupby("area")}})
    res = pd.DataFrame(res)
    res.to_csv(ROOT / "results" / "samtale_decoding_check.csv", index=False)
    pd.set_option("display.width", 200)
    print("\n" + (res.set_index("variant") * 100).round(1).to_string())
    print("\n(WER in %, change vs greedy with 95% CI over speakers; area columns are descriptive, few speakers per area)")

if __name__ == "__main__":
    main()
