import time

import pandas as pd
import torch

from dialect_probing.config import ROOT, load_config
from dialect_probing.weight_space import Audio, blend, decode, gap_ci, load_model, module3, score, set_top, summary

GROUPS = {
    "ff1": ("feed_forward1.",), "attn_qkv": ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
    "attn_post": ("self_attn.post",), "rel_pos": ("self_attn.relative_k_proj",), "per_dim_scale": ("self_attn.per_dim_scale",),
    "lconv": ("lconv1d.",), "ff2": ("feed_forward2.",), "norms": ("norm_pre_attn", "norm_post_attn", "norm_out"),
}
ALPHAS = (0.25, 0.5, 0.75, 1.25, 1.5, 2.0)

def main():
    cfg = load_config()
    out = cfg["out_dir"] / "weight_space"
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    model, units, ft, orig = load_model(cfg, device)
    audio = Audio(cfg)
    conds = {"fine-tuned": blend(ft, orig), "original": blend(ft, orig, 0.0)}
    conds |= {f"revert {g}": blend(ft, orig, revert=p) for g, p in GROUPS.items()}
    conds |= {f"delta x{a}": blend(ft, orig, a) for a in ALPHAS}
    rows, segs = [], []
    for half in ("tune", "test"):
        seg = module3(cfg, half)
        base = None
        for name, sd in conds.items():
            set_top(model, sd)
            t0 = time.time()
            sc = score(seg, decode(model, units, seg, audio, device))
            base = sc if base is None else base
            lo, hi = gap_ci(base, sc) if sc is not base else gap_ci(sc)
            s = summary(sc)
            rows.append({"half": half, "condition": name, **s, "ci_lo": lo, "ci_hi": hi, "ci_of": "gap" if sc is base else "gap change"})
            segs.append(sc.assign(half=half, condition=name))
            print(f"{half} {name:20s} all {100 * s['all']:5.1f} West {100 * s['west']:5.1f} East {100 * s['east']:5.1f} "
                  f"gap {100 * s['gap']:+5.1f} [{100 * lo:+.1f},{100 * hi:+.1f}] ({time.time() - t0:.0f} s)", flush=True)
    pd.concat(segs).to_parquet(out / "l12_revert_segments.parquet")
    pd.DataFrame(rows).to_csv(ROOT / "results" / "l12_revert.csv", index=False)

if __name__ == "__main__":
    main()
