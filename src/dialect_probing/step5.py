import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dialect_probing.written_standard import LEXICONS

MARKERS = {"jeg", "meg", "deg", "seg", "hun", "vi", "dere", "de", "dem", "ikke", "hva", "hvor", "hvordan", "noe", "noen", "mye"}
RETROFLEX = re.compile(r"r[tdnls]")

class Activations:
    def __init__(self, act_dir: Path):
        self.dir = Path(act_dir)
        self.meta = json.loads((self.dir / "meta.json").read_text())
        self.words = pd.read_parquet(self.dir / "words.parquet")
        self.words["sentence_id"] = self.words["utt_id"].str.rsplit("-", n=1).str[1]
        self.sweep = np.load(self.dir / "lag_sweep.npy", mmap_mode="r")
        self.ctc = np.load(self.dir / "ctc_shifted_ln.npy", mmap_mode="r")
        self.layers = self.meta["layers"]

    def vectors(self, rows: np.ndarray, layer: int, variant: str, shift: int = 0) -> np.ndarray:
        if variant == "ctc":
            return np.asarray(self.ctc[rows, layer], dtype=np.float32)
        return np.asarray(self.sweep[rows, layer, shift if variant == "lag" else 0], dtype=np.float32)

def speaker_split(speakers: pd.DataFrame, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    half = {}
    for _, g in speakers.groupby(["group", "gender"]):
        names = list(g["speaker"])
        rng.shuffle(names)
        for i, s in enumerate(names):
            half[s] = "select" if i % 2 == 0 else "analysis"
    return pd.Series(half, name="half")

def speaker_folds(speakers: pd.DataFrame, k: int, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    fold = {}
    for _, g in speakers.groupby("group"):
        names = list(g["speaker"])
        rng.shuffle(names)
        for i, s in enumerate(names):
            fold[s] = i % k
    return pd.Series(fold, name="fold")

def probe_accuracy(x_tr, y_tr, x_te, y_te, n_classes: int, device, weight_decay: float = 1e-3, steps: int = 300) -> float:
    x_tr = torch.as_tensor(x_tr, device=device)
    x_te = torch.as_tensor(x_te, device=device)
    mu, sd = x_tr.mean(0), x_tr.std(0) + 1e-6
    x_tr, x_te = (x_tr - mu) / sd, (x_te - mu) / sd
    y = torch.as_tensor(np.asarray(y_tr, dtype=np.int64), device=device)
    lin = torch.nn.Linear(x_tr.shape[1], n_classes).to(device)
    opt = torch.optim.Adam(lin.parameters(), lr=1e-2, weight_decay=weight_decay)
    for _ in range(steps):
        opt.zero_grad()
        torch.nn.functional.cross_entropy(lin(x_tr), y).backward()
        opt.step()
    with torch.no_grad():
        pred = lin(x_te).argmax(1).cpu().numpy()
    return float((pred == np.asarray(y_te)).mean())

def unit_rows(x: np.ndarray) -> np.ndarray:
    x = x - x.mean(0)
    return x / np.linalg.norm(x, axis=1, keepdims=True)

class PairStats:
    def __init__(self, x: np.ndarray, meta: pd.DataFrame, min_groups: int, exclude_same_sentence: bool = True):
        x = unit_rows(x)
        self.speaker_codes, self.speakers = pd.factorize(meta["speaker"])
        self.items = []
        for word, idx in meta.groupby("word").indices.items():
            g = meta["group"].to_numpy()[idx]
            if len(np.unique(g)) < min_groups:
                continue
            spk = self.speaker_codes[idx]
            sent = meta["sentence_id"].to_numpy()[idx]
            valid = spk[:, None] != spk[None, :]
            if exclude_same_sentence:
                valid &= sent[:, None] != sent[None, :]
            self.items.append((word, idx, x[idx] @ x[idx].T, valid, spk))

    def per_word(self, speaker_groups: np.ndarray | None = None, meta: pd.DataFrame | None = None) -> pd.DataFrame:
        rows = []
        for word, idx, gram, valid, spk in self.items:
            g = speaker_groups[spk] if speaker_groups is not None else meta["group"].to_numpy()[idx]
            same = g[:, None] == g[None, :]
            w, c = valid & same, valid & ~same
            if w.any() and c.any():
                rows.append({"word": word, "within": gram[w].mean(), "cross": gram[c].mean(), "tokens": len(idx)})
        return pd.DataFrame(rows)

    def to_reference(self, meta: pd.DataFrame, reference: list[str]) -> pd.DataFrame:
        rows = []
        for word, idx, gram, valid, _ in self.items:
            g = meta["group"].to_numpy()[idx]
            ref = np.isin(g, reference)
            for grp in np.unique(g):
                m = valid & (g == grp)[:, None] & (ref & (g != grp))[None, :]
                if m.any():
                    rows.append({"word": word, "group": grp, "sim": gram[m].mean()})
        return pd.DataFrame(rows)

def summarize(per_word: pd.DataFrame) -> dict:
    w, c = per_word["within"].mean(), per_word["cross"].mean()
    return {"within": w, "cross": c, "gap": w - c, "ratio": c / w, "types": len(per_word)}

def word_categories(words: pd.Series, ordbank_dir: Path) -> pd.Series:
    inf = set()
    with open(Path(ordbank_dir) / LEXICONS["nob"], encoding="latin-1") as f:
        next(f)
        for line in f:
            cols = line.split("\t")
            if len(cols) > 3 and cols[3].startswith("verb inf"):
                inf.add(cols[2].lower())

    def cat(w: str) -> str:
        if w in MARKERS:
            return "markers"
        if w in inf and w.endswith("e"):
            return "infinitive"
        if RETROFLEX.search(w):
            return "retroflex"
        if "r" in w:
            return "other r"
        return "control"

    return words.map(cat)
