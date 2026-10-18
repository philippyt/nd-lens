import json

import numpy as np
import pandas as pd
from norsk_asr.evaluate import score
from norsk_asr.lm import ctc_labels, make_decoder
from norsk_asr.units import Units

from dialect_probing.config import ROOT, load_config
from dialect_probing.m3_decode import greedy_words, lm_decode

SETS = ["fleurs_nb", "npsc_nb", "nst_val", "stortinget_val"]
VARIANTS = {"greedy": None, "lm as shipped (a 0.6, b 1)": (0.6, 1.0, 0.0), "lm retuned (a 0.4, b 2)": (0.4, 2.0, 0.0),
            "lm retuned + blank penalty 1.0": (0.4, 2.0, 1.0)}

def main():
    cfg = load_config()
    bench = cfg["out_dir"] / "bench"
    units = Units.load(cfg["checkpoint"] / "units.json")
    dec = make_decoder(str(cfg["norsk_asr_dir"] / "data" / "lm" / "no4"), ctc_labels(units), 0.6, 1.0)
    rows = []
    for name in SETS:
        z = np.load(bench / f"{name}.logprobs.npz")
        meta = [json.loads(line) for line in open(bench / f"{name}.jsonl")]
        lps = [z[m["id"]].astype(np.float32) for m in meta]
        refs, digits = [m["ref"] for m in meta], [m["has_digits"] for m in meta]
        for v, p in VARIANTS.items():
            if p is None:
                hyps = [" ".join(w for w, _ in greedy_words(lp, units.pieces)) for lp in lps]
            else:
                dec.reset_params(alpha=p[0], beta=p[1])
                hyps = [" ".join(w for w, _ in ws) for ws in lm_decode(dec, lps, list(range(len(lps))), p[2])]
            r = score(refs, hyps, digits)
            rows.append({"set": name, "variant": v, "utts": len(refs), "wer": r["wer"], "wer_nodigit": r.get("wer_nodigit", r["wer"])})
            print(f"  {name:15s} {v:32s} WER {100 * rows[-1]['wer_nodigit']:5.1f} (digit-free)  {100 * r['wer']:5.1f} (all)", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(ROOT / "results" / "benchmark_decoding.csv", index=False)
    t = (res.pivot(index="variant", columns="set", values="wer_nodigit")[SETS] * 100).reindex(list(VARIANTS))
    t["mean"] = t.mean(1)
    print("\nWER (%) on digit-free utterances, same subsets as the norsk-asr README\n" + t.round(1).to_string())

if __name__ == "__main__":
    main()
