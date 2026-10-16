import numpy as np
import torch

FRAME_SECONDS = 0.04

def ctc_viterbi(lp: np.ndarray, targets: list[int], blank: int = 0) -> tuple[np.ndarray, float] | None:
    t_len, n = lp.shape[0], len(targets)
    s_len = 2 * n + 1
    if n == 0 or t_len < n:
        return None
    ext = np.full(s_len, blank)
    ext[1::2] = targets
    emit = lp[:, ext]
    skip = np.zeros(s_len, bool)
    skip[3::2] = ext[3::2] != ext[1:-2:2]
    dp = np.full((t_len, s_len), -np.inf)
    bp = np.zeros((t_len, s_len), np.int8)
    dp[0, 0], dp[0, 1] = emit[0, 0], emit[0, 1]
    pad1, pad2 = np.full(1, -np.inf), np.full(2, -np.inf)
    cols = np.arange(s_len)
    for t in range(1, t_len):
        prev = dp[t - 1]
        cand = np.stack([prev, np.concatenate([pad1, prev[:-1]]), np.where(skip, np.concatenate([pad2, prev[:-2]]), -np.inf)])
        arg = cand.argmax(0)
        dp[t] = cand[arg, cols] + emit[t]
        bp[t] = arg
    end = s_len - 1 if dp[-1, -1] >= dp[-1, -2] else s_len - 2
    score = dp[-1, end]
    if not np.isfinite(score):
        return None
    path = np.empty(t_len, int)
    s = end
    for t in range(t_len - 1, -1, -1):
        path[t] = s
        s -= int(bp[t, s])
    return path, float(score)

def word_spans(path: np.ndarray, token_word: np.ndarray, n_words: int) -> list[tuple[int, int] | None]:
    spans: list[tuple[int, int] | None] = [None] * n_words
    for t, s in enumerate(path):
        if s % 2 == 0:
            continue
        w = token_word[(s - 1) // 2]
        spans[w] = (t, t + 1) if spans[w] is None else (spans[w][0], t + 1)
    return spans

@torch.no_grad()
def log_probs(model, audio: np.ndarray, device) -> np.ndarray:
    wav = torch.from_numpy(audio)[None].to(device)
    lens = torch.tensor([len(audio)], device=device)
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        lp, olens = model(wav, lens)
    return lp[0, : olens[0]].float().cpu().numpy()
