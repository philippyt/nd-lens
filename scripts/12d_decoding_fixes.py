import itertools
import time
from pathlib import Path

import kenlm
import numpy as np
import pandas as pd
from norsk_asr.lm import ctc_labels
from norsk_asr.units import Units
from pyctcdecode.alphabet import Alphabet
from pyctcdecode.decoder import BeamSearchDecoderCTC
from pyctcdecode.language_model import LanguageModel, MultiLanguageModel

from dialect_probing.config import ROOT, load_config
from dialect_probing.m3_decode import counts, gap_change_ci, gap_ci, greedy_words, lm_decode, load_logprobs

def lm(path: Path) -> LanguageModel:
    return LanguageModel(kenlm.Model(f"{path}.bin"), Path(f"{path}.unigrams.txt").read_text().split())

def pooled(c: pd.DataFrame, col: str = "err") -> float:
    return c[col].sum() / c["words"].sum()

def main():
    cfg = load_config()
    out = cfg["out_dir"] / "spont"
    units = Units.load(cfg["checkpoint"] / "units.json")
    ix = pd.read_parquet(out / "m3_logprobs_index.parquet").merge(pd.read_csv(out / "m3_split.csv")[["speaker", "half"]], on="speaker")
    lps = load_logprobs(out / "m3_logprobs.npz")
    halves = {h: ix.index[ix["half"] == h].tolist() for h in ("tune", "test")}
    base, conv = lm(cfg["norsk_asr_dir"] / "data" / "lm" / "no4"), lm(cfg["data_dir"] / "lm_conv" / "conv4")
    alphabet = Alphabet.build_alphabet(ctc_labels(units))
    dec_base, dec_mix = BeamSearchDecoderCTC(alphabet, base), BeamSearchDecoderCTC(alphabet, MultiLanguageModel([base, conv]))
    log = []

    def run(kind: str, half: str, a: float = 0.0, b: float = 0.0, bp: float = 0.0, ac: float = 0.0):
        idx = halves[half]
        t0 = time.time()
        if kind == "greedy":
            words = [greedy_words(lps[i], units.pieces, bp) for i in idx]
        elif kind == "lm":
            base.reset_params(alpha=a, beta=b)
            words = lm_decode(dec_base, lps, idx, bp)
        else:
            base.reset_params(alpha=2 * a, beta=2 * b)
            conv.reset_params(alpha=2 * ac, beta=0.0)
            words = lm_decode(dec_mix, lps, idx, bp)
        c = counts(ix.loc[idx], words)
        if half == "tune":
            log.append({"kind": kind, "alpha": a, "beta": b, "blank_penalty": bp, "alpha_conv": ac, "wer": pooled(c),
                        "del": pooled(c, "del"), "ins": pooled(c, "ins"), "seconds": time.time() - t0})
            print(f"  tune {kind:6s} a={a:.1f} b={b:.1f} bp={bp:.2f} ac={ac:.2f}  WER {100 * pooled(c):.2f}  del {100 * pooled(c, 'del'):.2f}  "
                  f"ins {100 * pooled(c, 'ins'):.2f}  ({time.time() - t0:.0f} s)", flush=True)
        return c

    def best(**fixed) -> dict:
        t = pd.DataFrame(log)
        for k, v in fixed.items():
            t = t[t[k] == v] if not callable(v) else t[v(t[k])]
        return t.loc[t["wer"].idxmin()].to_dict()

    chosen = {"greedy": {"kind": "greedy"}, "lm (shipped a=0.6 b=1.0)": {"kind": "lm", "a": 0.6, "b": 1.0}}
    for bp in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]:
        run("greedy", "tune", bp=bp)
    g = best(kind="greedy")
    chosen["greedy + blank penalty"] = {"kind": "greedy", "bp": g["blank_penalty"]}
    for a, b in itertools.product([0.4, 0.6, 0.8, 1.0], [0.0, 1.0, 2.0, 3.0, 4.0]):
        run("lm", "tune", a, b)
    l0 = best(kind="lm", blank_penalty=0.0)
    chosen["lm retuned a, b"] = {"kind": "lm", "a": l0["alpha"], "b": l0["beta"]}
    for a, b, bp in itertools.product([l0["alpha"] - 0.2, l0["alpha"], l0["alpha"] + 0.2], [l0["beta"], l0["beta"] + 1, l0["beta"] + 2, l0["beta"] + 3], [0.5, 1.0, 1.5, 2.0, 3.0]):
        run("lm", "tune", a, b, bp)
    l1 = best(kind="lm")
    chosen["lm retuned + blank penalty"] = {"kind": "lm", "a": l1["alpha"], "b": l1["beta"], "bp": l1["blank_penalty"]}
    for a, ac, b in itertools.product([l0["alpha"] - 0.2, l0["alpha"]], [0.1, 0.2, 0.4, 0.6], [l0["beta"], l0["beta"] + 1, l0["beta"] + 2]):
        run("mix", "tune", a, b, 0.0, ac)
    m0 = best(kind="mix", blank_penalty=0.0)
    chosen["lm + conversational lm"] = {"kind": "mix", "a": m0["alpha"], "b": m0["beta"], "ac": m0["alpha_conv"]}
    for b, bp in itertools.product([m0["beta"], m0["beta"] + 1, m0["beta"] + 2], [0.5, 1.0, 1.5, 2.0]):
        run("mix", "tune", m0["alpha"], b, bp, m0["alpha_conv"])
    m1 = best(kind="mix")
    chosen["lm + conversational lm + blank penalty"] = {"kind": "mix", "a": m1["alpha"], "b": m1["beta"], "bp": m1["blank_penalty"], "ac": m1["alpha_conv"]}
    pd.DataFrame(log).to_csv(out / "m3_decoding_tune.csv", index=False)
    rng = np.random.default_rng(0)
    rows, test = [], {}
    for name, p in chosen.items():
        c = test[name] = run(p["kind"], "test", p.get("a", 0.0), p.get("b", 0.0), p.get("bp", 0.0), p.get("ac", 0.0))
        w, e = c[c["area"] == "west"], c[c["area"] == "east"]
        gap, lo, hi = gap_ci(c, "err", rng)
        dgap, dlo, dhi = gap_ci(c, "del", rng)
        clo, chi, rlo, rhi, pneg = gap_change_ci(c, test["greedy"], rng)
        rows.append({"variant": name, **{k: v for k, v in p.items() if k != "kind"}, "wer_all": pooled(c), "wer_west": pooled(w),
                     "wer_east": pooled(e), "gap": gap, "gap_lo": lo, "gap_hi": hi, "rel_gap": pooled(w) / pooled(e) - 1,
                     "del_all": pooled(c, "del"), "del_west": pooled(w, "del"), "del_east": pooled(e, "del"), "del_gap": dgap,
                     "del_gap_lo": dlo, "del_gap_hi": dhi, "ins_all": pooled(c, "ins"),
                     "change_lo": clo, "change_hi": chi, "rel_change_lo": rlo, "rel_change_hi": rhi, "share_boot_smaller": pneg})
    res = pd.DataFrame(rows)
    res.to_csv(ROOT / "results" / "decoding_fixes.csv", index=False)
    print("\ntest half (speakers not used for tuning), 95% CI over speakers")
    base = res.iloc[0]
    for r in res.itertuples():
        pr = ", ".join(f"{k}={getattr(r, k):g}" for k in ("a", "b", "bp", "ac") if k in res and pd.notna(getattr(r, k)) and getattr(r, k))
        print(f"  {r.variant:40s} WER {100 * r.wer_all:5.1f}  West {100 * r.wer_west:5.1f}  East {100 * r.wer_east:5.1f}  "
              f"gap {100 * r.gap:+.1f} [{100 * r.gap_lo:+.1f}, {100 * r.gap_hi:+.1f}] ({r.rel_gap:+.0%})  "
              f"del {100 * r.del_all:4.1f} (W-E {100 * r.del_gap:+.1f} [{100 * r.del_gap_lo:+.1f}, {100 * r.del_gap_hi:+.1f}])  "
              f"ins {100 * r.ins_all:4.1f}  {pr}")
        if r.Index:
            print(f"  {'':40s} gap change vs greedy {100 * (r.gap - base.gap):+.1f} [{100 * r.change_lo:+.1f}, {100 * r.change_hi:+.1f}], "
                  f"West/East ratio change {r.rel_gap - base.rel_gap:+.2f} [{r.rel_change_lo:+.2f}, {r.rel_change_hi:+.2f}]")

if __name__ == "__main__":
    main()
