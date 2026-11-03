import io
import math

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open

from dialect_probing.config import ROOT, load_config

AREA = {"g06": "west", "g07": "west", "g08": "west", "g09": "west", "g11": "east", "g12": "east"}
INPUT = {"feed_forward1.ffw_layer_1.linear.weight": "feed_forward1.pre_layer_norm.weight",
         "self_attn.q_proj.linear.weight": "norm_pre_attn.weight", "self_attn.k_proj.linear.weight": "norm_pre_attn.weight",
         "self_attn.v_proj.linear.weight": "norm_pre_attn.weight", "lconv1d.linear_start.linear.weight": "lconv1d.pre_layer_norm.weight"}

def positions(n: int = 13, d: int = 1024) -> torch.Tensor:
    """Gemma4 audio relative position embeddings for distances 12..0 frames."""
    inv = torch.exp(torch.arange(d // 2) * -(math.log(10000.0) / (d // 2 - 1)))
    t = torch.arange(n - 1, -1, -1)[:, None] * inv
    return torch.cat([t.sin(), t.cos()], -1)

def main():
    cfg = load_config()
    ck = safe_open(str(cfg["checkpoint"] / "model.safetensors"), "pt")
    gg = safe_open(hf_hub_download("google/embeddinggemma-2", "model.safetensors"), "pt")
    ft = lambda k: ck.get_tensor("encoder.layers.11." + k).float()
    og = lambda k: gg.get_tensor("audio_tower.layers.11." + k).float()
    keys = sorted(k[len("encoder.layers.11."):] for k in ck.keys() if k.startswith("encoder.layers.11."))

    rows = []
    for k in keys:
        a, b = ft(k), og(k)
        d = a - b
        r = {"param": k, "shape": "x".join(map(str, a.shape)), "rel_change": (d.norm() / b.norm()).item()}
        if d.ndim == 2:
            sv = torch.linalg.svdvals(d)
            e = (sv ** 2).cumsum(0) / (sv ** 2).sum()
            r |= {"rank_50": int((e < 0.5).sum()) + 1, "rank_90": int((e < 0.9).sum()) + 1,
                  "effective_rank": (sv.sum() ** 2 / (sv ** 2).sum()).item()}
        rows.append(r)
    stats = pd.DataFrame(rows)
    stats.to_csv(ROOT / "results" / "l12_delta.csv", index=False)
    print(stats.round(4).to_string(index=False))

    k = "self_attn.relative_k_proj.weight"
    wf, wo = ft(k), og(k)
    d = wf - wo
    learned = d - ((d * wo).sum() / (wo * wo).sum()) * wo
    u, s, vh = torch.linalg.svd(learned)
    p = positions()
    print(f"\nrelative position projection: share of learned change in top component {(s[0] ** 2 / (s ** 2).sum()).item():.2f}")
    print("top component response by distance 12..0 frames:", np.round((p @ vh[0]).numpy(), 2).tolist())
    print("top component output energy per head:", np.round((u[:, 0].view(8, 128).norm(dim=1) ** 2).numpy(), 3).tolist())

    z = np.load(cfg["out_dir"] / "spont" / "enc_spont.npz")
    meta = pd.read_json(io.StringIO(str(z["meta"])))
    meta["area"] = meta["group"].map(AREA)
    keep = meta["area"].notna().to_numpy()
    x, m = torch.from_numpy(z["x"][keep, 11].astype(np.float32)), meta[keep].reset_index(drop=True)
    ok = m["word"].map(m.groupby("word")["area"].nunique()).eq(2).to_numpy()
    x, m = x[ok], m[ok].reset_index(drop=True)
    x = x - torch.from_numpy(pd.DataFrame(x.numpy()).groupby(m["word"].to_numpy()).transform("mean").to_numpy())
    spk = m.groupby("speaker")["area"].first()
    sm = torch.from_numpy(pd.DataFrame(x.numpy()).groupby(m["speaker"].to_numpy()).mean().loc[spk.index].to_numpy())
    print(f"\n{len(m)} tokens, {m['word'].nunique()} word types, {len(spk)} speakers (analysis half)")

    def direction(labels):
        v = sm[labels == "west"].mean(0) - sm[labels == "east"].mean(0)
        return v / v.norm()

    true = direction(spk.to_numpy())
    rng = np.random.default_rng(0)
    null = torch.stack([direction(rng.permutation(spk.to_numpy())) for _ in range(500)])
    rows = []
    for k, norm in INPUT.items():
        gain = ft(norm)
        for name, w in (("delta", ft(k) - og(k)), ("original", og(k))):
            f = lambda v: (w @ (v * gain).T).norm(dim=0)
            t, n = f(true[None])[0].item(), f(null)
            rows.append({"matrix": k.split(".linear")[0], "weights": name, "ratio_vs_shuffled": t / n.mean().item(),
                         "p": ((n >= t).sum().item() + 1) / (len(n) + 1)})
    al = pd.DataFrame(rows)
    al.to_csv(ROOT / "results" / "l12_delta_alignment.csv", index=False)
    print(al.round(3).to_string(index=False))

if __name__ == "__main__":
    main()
