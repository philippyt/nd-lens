import jiwer
import numpy as np
import pandas as pd
import torch
from norsk_asr.evaluate import greedy
from norsk_asr.text import normalize
from norsk_asr.train import load_checkpoint
from norsk_asr.transcribe import load_audio
from transformers import AutoModel

SR = 16000

def load_model(cfg, device):
    model, units = load_checkpoint(cfg["checkpoint"], device)
    google = AutoModel.from_pretrained(model.cfg.encoder_id, dtype=torch.float32).audio_tower
    top = len(model.encoder.layers) - 1
    orig = {k: v.to(device) for k, v in google.layers[top].state_dict().items()}
    del google
    ft = {k: v.detach().clone() for k, v in model.encoder.layers[top].state_dict().items()}
    return model, units, ft, orig

def module3(cfg, half: str | None = None) -> pd.DataFrame:
    seg = pd.read_parquet(cfg["data_dir"] / "spont" / "segments.parquet")
    seg = seg[seg["set"] == "nbtale_m3"].sort_values(["path", "start"]).reset_index(drop=True)
    seg["half"] = seg["speaker"].map(pd.read_csv(cfg["out_dir"] / "spont" / "m3_split.csv").set_index("speaker")["half"])
    return seg if half is None else seg[seg["half"] == half].reset_index(drop=True)

class Audio:
    """Module 3 segments packed as int16 in outputs/spont/m3_audio.npy (built from the wavs on first use)."""

    def __init__(self, cfg):
        path, ix = cfg["out_dir"] / "spont" / "m3_audio.npy", cfg["out_dir"] / "spont" / "m3_audio_index.parquet"
        if not path.exists():
            seg, chunks, off, cache = module3(cfg), [], [0], {}
            for r in seg.itertuples():
                if r.path not in cache:
                    cache = {r.path: load_audio(r.path)}
                x = cache[r.path][int(r.start * SR):int(np.ceil(r.end * SR))]
                chunks.append(np.clip(x * 32767, -32768, 32767).astype(np.int16))
                off.append(off[-1] + len(x))
            np.save(path, np.concatenate(chunks))
            pd.DataFrame({"seg_id": seg["seg_id"], "a0": off[:-1], "a1": off[1:]}).to_parquet(ix)
        self.x = np.load(path, mmap_mode="r")
        self.ix = pd.read_parquet(ix).set_index("seg_id")

    def __call__(self, seg_id: str) -> np.ndarray:
        a0, a1 = self.ix.loc[seg_id, ["a0", "a1"]]
        return np.asarray(self.x[a0:a1], dtype=np.float32) / 32767

def set_top(model, sd: dict):
    model.encoder.layers[len(model.encoder.layers) - 1].load_state_dict(sd)

def blend(ft: dict, orig: dict, alpha: float = 1.0, revert: tuple = ()) -> dict:
    out = {}
    for k in ft:
        a = 0.0 if any(k.startswith(p) for p in revert) else alpha
        out[k] = orig[k] + a * (ft[k] - orig[k]) if ft[k].is_floating_point() else ft[k]
    return out

@torch.no_grad()
def decode(model, units, seg: pd.DataFrame, audio: Audio, device, batch_seconds: float = 600, return_lp: bool = False):
    waves = [audio(s) for s in seg["seg_id"]]
    order = np.argsort([len(w) for w in waves])
    hyps, lps, i = [None] * len(seg), [None] * len(seg), 0
    while i < len(order):
        j = i + 1
        while j < len(order) and len(waves[order[j]]) * (j - i + 1) <= batch_seconds * SR:
            j += 1
        idx = order[i:j]
        lens = torch.tensor([len(waves[k]) for k in idx])
        x = torch.zeros(len(idx), int(lens.max()))
        for b, k in enumerate(idx):
            x[b, :lens[b]] = torch.from_numpy(waves[k])
        with torch.autocast(device.type, dtype=torch.bfloat16):
            lp, ol = model(x.to(device), lens.to(device))
        for b, (k, ids) in enumerate(zip(idx, greedy(lp, ol))):
            hyps[k] = units.decode(ids)
            if return_lp:
                lps[k] = lp[b, :ol[b]].half().cpu().numpy()
        i = j
    return (hyps, lps) if return_lp else hyps

def score(seg: pd.DataFrame, hyps: list[str]) -> pd.DataFrame:
    rows = []
    for ref, h in zip(seg["ref"], hyps):
        o = jiwer.process_words(ref, " ".join(normalize(h or "").split()) or "<empty>")
        rows.append((o.hits + o.substitutions + o.deletions, o.substitutions + o.deletions + o.insertions,
                     o.substitutions, o.deletions, o.insertions))
    return seg[["seg_id", "speaker", "area", "group"]].assign(**dict(zip(["words", "err", "sub", "del", "ins"], np.array(rows).T)))

def summary(sc: pd.DataFrame) -> dict:
    a = sc.groupby("area")[["err", "words"]].sum()
    w = (a["err"] / a["words"]).to_dict()
    w["all"] = sc["err"].sum() / sc["words"].sum()
    w["macro"] = float(np.mean([w[x] for x in ("west", "east", "north", "central", "south")]))
    w["gap"] = w["west"] - w["east"]
    return w

def gap_ci(base: pd.DataFrame, new: pd.DataFrame | None = None, b: int = 2000, seed: int = 0) -> np.ndarray:
    """95% speaker-bootstrap CI of the West-East gap, or of its change from base to new on the same segments."""
    rng = np.random.default_rng(seed)
    per = lambda sc: sc.groupby(["area", "speaker"])[["err", "words"]].sum()
    A, B = per(base), per(new) if new is not None else None
    res = []
    for _ in range(b):
        g = {}
        for ar in ("west", "east"):
            sp = A.loc[ar].index.to_numpy()
            pick = rng.choice(sp, len(sp))
            ea = A.loc[ar].loc[pick].sum()
            v = ea["err"] / ea["words"]
            if B is not None:
                eb = B.loc[ar].loc[pick].sum()
                v = eb["err"] / eb["words"] - v
            g[ar] = v
        res.append(g["west"] - g["east"])
    return np.percentile(res, [2.5, 97.5])
