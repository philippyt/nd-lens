import io
import json
import time

import numpy as np
import pandas as pd
import torch
from norsk_asr.train import load_checkpoint
from norsk_asr.transcribe import load_audio

from dialect_probing.align import ctc_viterbi, log_probs, word_spans
from dialect_probing.config import load_config
from dialect_probing.nbtale import wav_path
from dialect_probing.spont_encoder import LAYERS, convergence, pool, states

SR = 16000

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
            audio = load_audio(str(wav_path(cfg["data_dir"] / "nbtale" / "raw", uid)))
            normed, lp = states(model, audio, device), log_probs(model, audio, device)
            targets = [t for x in g["word"] for t in units.encode(x)]
            res = ctc_viterbi(lp, targets)
            labels = np.array([0 if s % 2 == 0 else targets[s // 2] for s in res[0]]) if res else None
            for pos, r in enumerate(g.itertuples()):
                c0, c1 = int(r.ctc_start_frame), int(r.ctc_end_frame)
                vecs.append(pool(normed, c0, c1, lag, lags))
                score = float(lp[np.arange(c0, c1), labels[c0:c1]].mean()) if labels is not None and c1 > c0 else np.nan
                meta.append({"word": r.word, "speaker": r.speaker, "group": r.group, "sentence_id": uid.rsplit("-", 1)[1],
                             "unit": uid, "pos": pos, "c0": c0 - lag, "c1": c1 - lag, "align_score": score,
                             "ref_start": r.ref_start, "ref_end": r.ref_end})
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
            targets = [t for x in ids for t in x]
            token_word = np.concatenate([[i] * len(x) for i, x in enumerate(ids)]).astype(int)
            res = ctc_viterbi(lp, targets)
            if res is None:
                failed += 1
                continue
            path = res[0]
            labels = np.array([0 if s % 2 == 0 else targets[s // 2] for s in path])
            for wi, span in enumerate(word_spans(path, token_word, len(words))):
                if span is None:
                    continue
                vecs.append(pool(normed, span[0], span[1], lag, lags))
                score = float(lp[np.arange(span[0], span[1]), labels[span[0]:span[1]]].mean())
                meta.append({"word": words[wi], "speaker": r.speaker, "group": r.group, "sentence_id": r.seg_id,
                             "unit": r.seg_id, "pos": wi, "c0": span[0], "c1": span[1], "align_score": score})
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
