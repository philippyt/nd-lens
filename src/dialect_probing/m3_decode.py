from multiprocessing import get_context

import jiwer
import numpy as np
import pandas as pd
from norsk_asr.text import normalize

SPACE = "▁"
B = 10000

def split_speakers(seg: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for area, g in seg.groupby("area"):
        spk = sorted(g["speaker"].unique())
        spk = [spk[i] for i in rng.permutation(len(spk))]
        rows += [{"speaker": s, "area": area, "half": "tune" if i < len(spk) // 2 else "test"} for i, s in enumerate(spk)]
    return pd.DataFrame(rows)

def load_logprobs(path) -> list[np.ndarray]:
    z = np.load(path)
    return [z[f"s{i}"].astype(np.float32) for i in range(len(z.files))]

def greedy_words(lp: np.ndarray, pieces: list[str], blank_penalty: float = 0.0) -> list[tuple[str, tuple[int, int]]]:
    x = lp.copy()
    x[:, 0] -= blank_penalty
    best = x.argmax(1)
    words, cur, start, prev = [], "", 0, -1
    for t, k in enumerate(best):
        if k == prev or k == 0:
            prev = k
            continue
        prev = k
        p = pieces[k]
        if p.startswith(SPACE):
            if cur:
                words.append((cur, (start, last)))
            cur, start = p[1:], t
        else:
            if not cur:
                start = t
            cur += p
        last = t + 1
    if cur:
        words.append((cur, (start, last)))
    return words

def word_confidence(lp: np.ndarray, span: tuple[int, int]) -> tuple[float, float]:
    s, e = span
    e = max(e, s + 1)
    w = lp[s:e]
    emit = w.argmax(1) != 0
    v = w[emit].max(1) if emit.any() else np.array([w[:, 1:].max()])
    return float(v.mean()), float(v.min())

def expand(words: list[tuple[str, tuple[int, int]]]) -> tuple[list[str], list[tuple[int, int]]]:
    toks, spans = [], []
    for w, sp in words:
        for n in normalize(w).split():
            toks.append(n)
            spans.append(sp)
    return toks, spans

def align(ref: str, hyp: list[str]) -> tuple[np.ndarray, dict, np.ndarray]:
    lab = np.zeros(len(hyp), int)
    dels = np.zeros(len(hyp) + 1, int)
    c = {"words": len(ref.split()), "sub": 0, "del": 0, "ins": 0}
    if not hyp:
        dels[0] = c["words"]
        c["del"] = c["words"]
        return lab, c, dels
    for a in jiwer.process_words(ref, " ".join(hyp)).alignments[0]:
        if a.type == "substitute":
            lab[a.hyp_start_idx:a.hyp_end_idx] = 1
            c["sub"] += a.hyp_end_idx - a.hyp_start_idx
        elif a.type == "insert":
            lab[a.hyp_start_idx:a.hyp_end_idx] = 1
            c["ins"] += a.hyp_end_idx - a.hyp_start_idx
        elif a.type == "delete":
            dels[a.hyp_start_idx] += a.ref_end_idx - a.ref_start_idx
            c["del"] += a.ref_end_idx - a.ref_start_idx
    return lab, c, dels

def gap_scores(lp: np.ndarray, spans: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    speech = 1 - np.exp(lp[:, 0])
    total, peak = [], []
    for s, e in zip([0] + [e for _, e in spans], [s for s, _ in spans] + [len(lp)]):
        seg = speech[s:e] if e > s else np.zeros(1)
        total.append(seg.sum())
        peak.append(seg.max())
    return np.array(total), np.array(peak)

_DEC = {}

def _beams(i: int):
    dec, lps, bp, beam = _DEC["dec"], _DEC["lps"], _DEC["bp"], _DEC["beam"]
    x = lps[i].copy()
    x[:, 0] -= bp
    b = dec.decode_beams(x, beam_width=beam)[0]
    return [(w, (int(s), int(e))) for w, (s, e) in b[2]]

def lm_decode(dec, lps: list[np.ndarray], idx: list[int], blank_penalty: float = 0.0, beam: int = 64, workers: int = 8):
    _DEC.update(dec=dec, lps=lps, bp=blank_penalty, beam=beam)
    with get_context("fork").Pool(workers) as pool:
        return pool.map(_beams, idx, chunksize=4)

def counts(seg: pd.DataFrame, words: list) -> pd.DataFrame:
    rows = []
    for r, w in zip(seg.itertuples(), words):
        toks, _ = expand(w)
        _, c, _ = align(r.ref, toks)
        rows.append({"speaker": r.speaker, "area": r.area, **c})
    d = pd.DataFrame(rows)
    d["err"] = d["sub"] + d["del"] + d["ins"]
    return d.groupby(["area", "speaker"])[["words", "err", "sub", "del", "ins"]].sum().reset_index()

def gap_ci(sp: pd.DataFrame, num: str, rng, a: str = "west", b: str = "east", den: str = "words") -> tuple[float, float, float]:
    w, e = sp[sp["area"] == a], sp[sp["area"] == b]
    bw = w[[num, den]].to_numpy(float)[rng.integers(0, len(w), (B, len(w)))].sum(1)
    be = e[[num, den]].to_numpy(float)[rng.integers(0, len(e), (B, len(e)))].sum(1)
    d = bw[:, 0] / bw[:, 1] - be[:, 0] / be[:, 1]
    obs = w[num].sum() / w[den].sum() - e[num].sum() / e[den].sum()
    return obs, *np.percentile(d, [2.5, 97.5])

def gap_change_ci(c: pd.DataFrame, base: pd.DataFrame, rng, num: str = "err") -> tuple[float, float, float, float, float]:
    m = c.merge(base, on=["area", "speaker"], suffixes=("", "_b"))
    out = []
    for area in ("west", "east"):
        a = m[m["area"] == area][[num, "words", f"{num}_b", "words_b"]].to_numpy(float)
        out.append(a[rng.integers(0, len(a), (B, len(a)))].sum(1))
    (w, e) = out
    d = (w[:, 0] / w[:, 1] - e[:, 0] / e[:, 1]) - (w[:, 2] / w[:, 3] - e[:, 2] / e[:, 3])
    r = (w[:, 0] / w[:, 1]) / (e[:, 0] / e[:, 1]) - (w[:, 2] / w[:, 3]) / (e[:, 2] / e[:, 3])
    return *np.percentile(d, [2.5, 97.5]), *np.percentile(r, [2.5, 97.5]), float((d < 0).mean())
