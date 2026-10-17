import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from dialect_probing.align import clip_window
from dialect_probing.step5 import PairStats, summarize

LAYERS = [f"L{i}" for i in range(13)]
WEST = ["g06", "g07", "g08", "g09"]

@torch.no_grad()
def states(model, audio: np.ndarray, device) -> torch.Tensor:
    wav = torch.from_numpy(audio)[None].to(device)
    feats, fmask = model.features(wav, torch.tensor([len(audio)], device=device))
    with torch.autocast(device.type, dtype=torch.bfloat16):
        out = model.encoder(feats, torch.ones_like(fmask), output_hidden_states=True, return_dict=True)
    hs = torch.stack(out.hidden_states)[:, 0].float()
    hs = hs[:, :int(model.output_mask(fmask.sum(1), hs.shape[1]).sum())]
    return F.layer_norm(hs, hs.shape[-1:])

def pool(normed: torch.Tensor, c0: int, c1: int, lag: int, lags: list[int]) -> np.ndarray:
    t = normed.shape[1]
    return torch.stack([normed[l, slice(*clip_window(c0 - lag + s, c1 - lag + s, t))].mean(0) for l, s in enumerate(lags)]).half().cpu().numpy()

def convergence(x: np.ndarray, meta: pd.DataFrame, min_groups: int, perms: int, east: list[str], types=None) -> list[dict]:
    rng = np.random.default_rng(0)
    if types is not None:
        keep = meta["word"].isin(types).to_numpy()
        x, meta = x[keep], meta[keep].reset_index(drop=True)
    res = []
    for l, name in enumerate(LAYERS):
        st = PairStats(x[:, l].astype(np.float32), meta, min_groups)
        s = summarize(st.per_word(meta=meta))
        groups = meta.drop_duplicates("speaker").set_index("speaker")["group"].reindex(st.speakers).to_numpy()
        null = np.array([summarize(st.per_word(speaker_groups=rng.permutation(groups)))["gap"] for _ in range(perms)])
        ref = st.to_reference(meta, east)
        d = 1 - ref.groupby("group")["sim"].mean()
        res.append({"layer": name, "layer_idx": l, **s, "p_perm": (1 + (null >= s["gap"]).sum()) / (1 + perms),
                    "dist_east_west": d[d.index.isin(WEST)].mean(), "dist_east_east": d[d.index.isin(east)].mean(),
                    "dist_east_others": d[~d.index.isin(east)].mean()})
    return res
