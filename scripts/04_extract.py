import copy
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from norsk_asr.train import load_checkpoint
from norsk_asr.transcribe import load_audio
from transformers import AutoModel

from dialect_probing.align import FRAME_SECONDS, clip_window, frame_window
from dialect_probing.config import load_config
from dialect_probing.nbtale import wav_path

LAYER_NAMES = [f"L{i}" for i in range(13)] + ["L12_orig"]
SWEEP_LAYERS = list(range(len(LAYER_NAMES)))
SWEEP_SHIFTS = list(range(11))
VERSIONS = ["finetuned"] * 13 + ["original"]

def original_encoder(model):
    enc = copy.deepcopy(model.encoder)
    google = AutoModel.from_pretrained(model.cfg.encoder_id, dtype=torch.float32).audio_tower
    top = len(enc.layers) - 1
    enc.layers[top].load_state_dict(google.layers[top].state_dict())
    del google
    return enc

@torch.no_grad()
def hidden_states(model, encoders, audio: np.ndarray, device) -> list[torch.Tensor]:
    wav = torch.from_numpy(audio)[None].to(device)
    feats, fmask = model.features(wav, torch.tensor([len(audio)], device=device))
    outs = []
    with torch.autocast(device.type, dtype=torch.bfloat16):
        for enc in encoders:
            out = enc(feats, torch.ones_like(fmask), output_hidden_states=True, return_dict=True)
            outs.append(torch.stack(out.hidden_states)[:, 0].float())
    n = int(model.output_mask(fmask.sum(1), outs[0].shape[1]).sum())
    return [o[:, :n] for o in outs]

def main():
    cfg = load_config()
    out = cfg["out_dir"] / "activations"
    out.mkdir(parents=True, exist_ok=True)
    words = pd.read_parquet(cfg["out_dir"] / "align" / "words.parquet").sort_values(["utt_id", "word_idx"]).reset_index(drop=True)
    mid = (words["ctc_start"] + words["ctc_end"] - words["ref_start"] - words["ref_end"]) / 2
    lag = int(round(mid.median() / FRAME_SECONDS))
    print(f"{len(words)} words, CTC midpoint lag {mid.median():.3f} s -> shift CTC windows back {lag} frames")

    n, d = len(words), 1024
    manual = np.lib.format.open_memmap(out / "manual.npy", "w+", np.float32, (n, len(LAYER_NAMES), d))
    ctc = np.lib.format.open_memmap(out / "ctc_shifted.npy", "w+", np.float32, (n, len(LAYER_NAMES), d))
    sweep = np.lib.format.open_memmap(out / "lag_sweep.npy", "w+", np.float16, (n, len(SWEEP_LAYERS), len(SWEEP_SHIFTS), d))
    ctc_ln = np.lib.format.open_memmap(out / "ctc_shifted_ln.npy", "w+", np.float16, (n, len(LAYER_NAMES), d))

    device = torch.device("cuda")
    model, _ = load_checkpoint(cfg["checkpoint"], device)
    encoders = [model.encoder, original_encoder(model).to(device).eval()]
    windows, max_diff, peak = [], 0.0, 0.0
    t0 = time.time()
    for uid, w in words.groupby("utt_id", sort=False):
        audio = load_audio(str(wav_path(cfg["data_dir"] / "nbtale" / "raw", uid)))
        ft, orig = hidden_states(model, encoders, audio, device)
        max_diff = max(max_diff, (ft[:12] - orig[:12]).abs().max().item())
        peak = max(peak, ft.abs().max().item(), orig[12].abs().max().item())
        states = torch.cat([ft, orig[12:13]])
        normed = F.layer_norm(states, (d,))
        t = states.shape[1]
        for i, r in w.iterrows():
            m0, m1 = frame_window(r["ref_start"], r["ref_end"], t)
            c0, c1 = clip_window(int(r["ctc_start_frame"]) - lag, int(r["ctc_end_frame"]) - lag, t)
            manual[i] = states[:, m0:m1].mean(1).cpu().numpy()
            ctc[i] = states[:, c0:c1].mean(1).cpu().numpy()
            ctc_ln[i] = normed[:, c0:c1].mean(1).half().cpu().numpy()
            sweep[i] = torch.stack([normed[:, slice(*clip_window(m0 + s, m1 + s, t))].mean(1) for s in SWEEP_SHIFTS], 1).half().cpu().numpy()
            windows.append({"manual_j0": m0, "manual_j1": m1, "ctc_j0": c0, "ctc_j1": c1})
    took = time.time() - t0
    for a in (manual, ctc, sweep, ctc_ln):
        a.flush()

    meta = words.join(pd.DataFrame(windows))
    meta.to_parquet(out / "words.parquet")
    (out / "meta.json").write_text(json.dumps({
        "layers": LAYER_NAMES, "versions": VERSIONS, "frame_seconds": FRAME_SECONDS, "ctc_lag_frames": lag,
        "sweep_layers": SWEEP_LAYERS, "sweep_shifts_frames": SWEEP_SHIFTS, "precision": "bf16 autocast",
        "max_abs_diff_layers_0_11_finetuned_vs_original": max_diff, "max_abs_activation": peak,
    }, indent=1))
    print(f"{meta['utt_id'].nunique()} utterances in {took:.0f} s")
    print(f"layers 0-11 fine-tuned vs original encoder: max abs diff {max_diff:.3g}")
    print(f"largest activation {peak:.4g}")
    print(f"manual window frames: median {(meta['manual_j1'] - meta['manual_j0']).median():.0f}; "
          f"CTC-shifted: median {(meta['ctc_j1'] - meta['ctc_j0']).median():.0f}")
    overlap = (np.minimum(meta["manual_j1"], meta["ctc_j1"]) - np.maximum(meta["manual_j0"], meta["ctc_j0"])).clip(lower=0)
    print(f"manual window covered by shifted CTC window: mean {(overlap / (meta['manual_j1'] - meta['manual_j0'])).mean():.2f}")
    print(f"pooled vectors finite: manual {np.isfinite(manual).all()}, ctc {np.isfinite(ctc).all()}, sweep {np.isfinite(sweep).all()}")

if __name__ == "__main__":
    main()
