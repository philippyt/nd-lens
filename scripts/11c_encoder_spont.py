import io
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from norsk_asr.train import load_checkpoint
from norsk_asr.transcribe import load_audio

from dialect_probing.align import clip_window, ctc_viterbi, log_probs, word_spans
from dialect_probing.config import load_config
from dialect_probing.nbtale import wav_path
from dialect_probing.step5 import PairStats, summarize

LAYERS = [f"L{i}" for i in range(13)]
SR = 16000
WEST = ["g06", "g07", "g08", "g09"]

@torch.no_grad()
def states(model, audio: np.ndarray, device):
    wav = torch.from_numpy(audio)[None].to(device)
    lens = torch.tensor([len(audio)], device=device)
    feats, fmask = model.features(wav, lens)
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

def main():
    cfg = load_config()
    out = cfg["out_dir"] / "spont"
    s5 = cfg["out_dir"] / "step5"
    act = cfg["out_dir"] / "activations"
    lag = json.loads((act / "meta.json").read_text())["ctc_lag_frames"]
    sel = json.loads((s5 / "selected_lags.json").read_text())
    lags = [sel[n] for n in LAYERS]
    half = pd.read_csv(s5 / "split.csv").set_index("speaker")["half"]
    east, excl = cfg["analysis"]["east_groups"], cfg["analysis"]["exclude_groups"]
    device = torch.device("cuda")
    model, units = load_checkpoint(cfg["checkpoint"], device)
    model.eval()

    sets = {}
    path_r, path_s = out / "enc_read.npz", out / "enc_spont.npz"
    if not path_r.exists():
        w = pd.read_parquet(cfg["out_dir"] / "align" / "words.parquet")
        w = w[(w["speaker"].map(half) == "analysis") & ~w["group"].isin(excl)].sort_values(["utt_id", "word_idx"])
        vecs, meta, t0 = [], [], time.time()
        for uid, g in w.groupby("utt_id", sort=False):
            normed = states(model, load_audio(str(wav_path(cfg["data_dir"] / "nbtale" / "raw", uid))), device)
            for r in g.itertuples():
                vecs.append(pool(normed, int(r.ctc_start_frame), int(r.ctc_end_frame), lag, lags))
                meta.append({"word": r.word, "speaker": r.speaker, "group": r.group, "sentence_id": uid.rsplit("-", 1)[1]})
        np.savez(path_r, x=np.stack(vecs), meta=pd.DataFrame(meta).to_json())
        print(f"read: {len(vecs)} words in {time.time() - t0:.0f} s", flush=True)
    if not path_s.exists():
        seg = pd.read_parquet(cfg["data_dir"] / "spont" / "segments.parquet")
        seg = seg[(seg["set"] == "nbtale_m3") & (seg["speaker"].map(half) == "analysis") & ~seg["group"].isin(excl) & seg["group"].notna()]
        vecs, meta, t0, failed, cache = [], [], time.time(), 0, {}
        for r in seg.sort_values(["path", "start"]).itertuples():
            if r.path not in cache:
                cache = {r.path: load_audio(r.path)}
            audio = cache[r.path][int(r.start * SR):int(np.ceil(r.end * SR))]
            normed, lp = states(model, audio, device), log_probs(model, audio, device)
            words = r.ref.split()
            ids = [units.encode(x) for x in words]
            if any(len(i) == 0 for i in ids):
                failed += 1
                continue
            token_word = np.concatenate([[i] * len(x) for i, x in enumerate(ids)]).astype(int)
            res = ctc_viterbi(lp, [t for x in ids for t in x])
            if res is None:
                failed += 1
                continue
            for wi, span in enumerate(word_spans(res[0], token_word, len(words))):
                if span is None:
                    continue
                vecs.append(pool(normed, span[0], span[1], lag, lags))
                meta.append({"word": words[wi], "speaker": r.speaker, "group": r.group, "sentence_id": r.seg_id})
        np.savez(path_s, x=np.stack(vecs), meta=pd.DataFrame(meta).to_json())
        print(f"spontaneous: {len(vecs)} words in {time.time() - t0:.0f} s; segments not aligned {failed}", flush=True)
    for name, p in (("read", path_r), ("spontaneous", path_s)):
        z = np.load(p, allow_pickle=True)
        sets[name] = (z["x"], pd.read_json(io.StringIO(str(z["meta"]))))

    mg = cfg["analysis"]["min_dialect_groups_per_word"]
    types = {n: {w for w, g in m.groupby("word") if g["group"].nunique() >= mg} for n, (x, m) in sets.items()}
    common = types["read"] & types["spontaneous"]
    print(f"word types in >= {mg} groups: read {len(types['read'])}, spontaneous {len(types['spontaneous'])}, common {len(common)}")
    rows = []
    for name, (x, m) in sets.items():
        for scope, ty in (("all types", None), ("common types", common)):
            for r in convergence(x, m, mg, 100, east, ty):
                rows.append({"set": name, "scope": scope, **r})
    res = pd.DataFrame(rows)
    res.to_csv(out / "encoder_read_vs_spont.csv", index=False)
    for scope, g in res.groupby("scope"):
        print(f"\n{scope}: across/within ratio (gap, perm p) and distance from East (West groups vs East)")
        p = g.pivot(index="layer_idx", columns="set", values=["ratio", "gap", "p_perm", "dist_east_west", "dist_east_east"])
        for l in range(13):
            r = p.loc[l]
            print(f"  L{l:<2d} read {r[('ratio', 'read')]:.3f} (gap {r[('gap', 'read')]:.3f}, p {r[('p_perm', 'read')]:.2f}) "
                  f"| spont {r[('ratio', 'spontaneous')]:.3f} (gap {r[('gap', 'spontaneous')]:.3f}, p {r[('p_perm', 'spontaneous')]:.2f}) "
                  f"| dist West-East minus East-East: read {r[('dist_east_west', 'read')] - r[('dist_east_east', 'read')]:+.3f}, "
                  f"spont {r[('dist_east_west', 'spontaneous')] - r[('dist_east_east', 'spontaneous')]:+.3f}")
        print("  types:", g.groupby("set")["types"].first().to_dict())

if __name__ == "__main__":
    main()
