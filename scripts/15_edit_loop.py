"""Hand-written weight edits, kept only if they help on the tune half.

    python scripts/15_edit_loop.py read                   # error profile of the current stack (tune half)
    python scripts/15_edit_loop.py try edits/03_x.py      # score a candidate on top of the accepted stack
    python scripts/15_edit_loop.py try edits/03_x.py --accept
    python scripts/15_edit_loop.py probe så               # what the head does where a word is kept or deleted
    python scripts/15_edit_loop.py final                  # accepted stack on the test half, once

An edit file defines apply(model, units) and changes weights in place, with no gradients and no search.
Target: mean WER over the five areas (tune half). Accept only if it drops by at least 0.1 points and West does
not get worse. Every scored pass is logged to outputs/edit_loop/log.jsonl.
"""
import argparse
import collections
import importlib.util
import json
import time
from pathlib import Path

import jiwer
import numpy as np
import pandas as pd
import torch
from norsk_asr.text import normalize

from dialect_probing.align import ctc_viterbi, word_spans
from dialect_probing.config import ROOT, load_config
from dialect_probing.weight_space import Audio, decode, gap_ci, load_model, module3, score, summary

MIN_GAIN = 0.001
AREAS = ["west", "east", "north", "central", "south"]

def load_edit(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def stack(state: dict) -> list[Path]:
    return [ROOT / p for p in state["accepted"]]

def build(cfg, device, edits: list[Path]):
    model, units, _, _ = load_model(cfg, device)
    with torch.no_grad():
        for p in edits:
            load_edit(p).apply(model, units)
    return model, units

def line(name: str, s: dict) -> str:
    return (f"{name:34s} mean {100 * s['macro']:5.2f} all {100 * s['all']:5.2f} " +
            " ".join(f"{a[:2]} {100 * s[a]:5.1f}" for a in AREAS) + f" gap {100 * s['gap']:+5.2f}")

def profile(seg: pd.DataFrame, hyps: list[str], sc: pd.DataFrame):
    t = sc.groupby("area")[["words", "sub", "del", "ins"]].sum()
    print("per 100 ref words:\n" + (t[["sub", "del", "ins"]].div(t["words"], axis=0) * 100).loc[AREAS].round(2).to_string())
    c = collections.defaultdict(collections.Counter)
    for r, h in zip(seg.itertuples(), hyps):
        h = " ".join(normalize(h or "").split()) or "<empty>"
        ref, hw = r.ref.split(), h.split()
        for a in jiwer.process_words(r.ref, h).alignments[0]:
            rs, hs = ref[a.ref_start_idx:a.ref_end_idx], hw[a.hyp_start_idx:a.hyp_end_idx]
            if a.type == "substitute":
                c[r.area].update(zip(rs, hs))
            elif a.type == "delete":
                c[r.area].update((w, "<del>") for w in rs)
            elif a.type == "insert":
                c[r.area].update(("<ins>", w) for w in hs)
    tot = sum(c.values(), collections.Counter())
    print("\nmost frequent word errors (all areas, count; West share):")
    for (rw, hw), n in tot.most_common(25):
        print(f"  {rw:>12s} -> {hw:<12s} {n:4d}  West {c['west'][(rw, hw)] / n:.2f}")

@torch.no_grad()
def probe(model, units, seg, audio, device, word: str, out: Path):
    """Force-align `word` in every reference; per occurrence: kept or deleted, what the head emits, and the
    reader's final hidden state (input to head_proj) pooled over the word's frames."""
    grab = {}
    hook = model.head_proj.register_forward_hook(lambda m, i, o: grab.__setitem__("h", i[0][0].float()))
    first = units.encode(word)[0]
    rows, vecs = [], []
    for r in seg[seg["ref"].str.contains(rf"\b{word}\b")].itertuples():
        w = torch.from_numpy(audio(r.seg_id))[None].to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16):
            lp, ol = model(w, torch.tensor([w.shape[1]], device=device))
        lp, h = lp[0, :ol[0]].float(), grab["h"][:ol[0]]
        hyp = " ".join(normalize(units.decode(greedy_ids(lp))).split()) or "<empty>"
        ref = r.ref.split()
        status = ["?"] * len(ref)
        for a in jiwer.process_words(r.ref, hyp).alignments[0]:
            status[a.ref_start_idx:a.ref_end_idx] = [a.type] * (a.ref_end_idx - a.ref_start_idx)
        targets, tw = [], []
        for wi, x in enumerate(ref):
            ids = units.encode(x)
            targets += ids
            tw += [wi] * len(ids)
        res = ctc_viterbi(lp.cpu().numpy(), targets)
        if res is None:
            continue
        spans = word_spans(res[0], np.array(tw), len(ref))
        for wi, x in enumerate(ref):
            if x != word or not spans[wi]:
                continue
            a0, a1 = spans[wi]
            win = lp[max(0, a0 - 2):a1 + 2]
            nb = win[:, 1:].max(0).values
            rows.append({"area": r.area, "status": status[wi], "pos": wi, "first_of_segment": wi == 0,
                         "prev": ref[wi - 1] if wi else "<s>", "next": ref[wi + 1] if wi + 1 < len(ref) else "</s>",
                         "max_p_nonblank": float(nb.max().exp()), "top_nonblank": units.pieces[int(nb.argmax()) + 1],
                         "margin_first": float((win[:, first] - win[:, 0]).max())})
            vecs.append(h[a0:a1].mean(0).cpu())
    hook.remove()
    d = pd.DataFrame(rows)
    d.to_csv(out / f"probe_{word}.csv", index=False)
    torch.save(torch.stack(vecs), out / f"probe_{word}.pt")
    print(f"'{word}': {len(d)} aligned occurrences")
    print(d.groupby(["area", "status"]).size().unstack(fill_value=0).loc[[x for x in AREAS if x in set(d["area"])]].to_string())
    print("\nby outcome: median max p(non-blank), median margin of first piece vs blank, share segment-initial")
    print(d.groupby("status").agg(n=("pos", "size"), p=("max_p_nonblank", "median"), margin=("margin_first", "median"),
                                  initial=("first_of_segment", "mean")).round(2).to_string())
    dd = d[d["status"] == "delete"]
    print("\ndeleted: top non-blank piece", collections.Counter(dd["top_nonblank"]).most_common(6))
    print("deleted: previous word", collections.Counter(dd["prev"]).most_common(8))
    print("kept:    previous word", collections.Counter(d[d["status"] == "equal"]["prev"]).most_common(8))

def greedy_ids(lp: torch.Tensor) -> list[int]:
    seq = torch.unique_consecutive(lp.argmax(-1)).tolist()
    return [u for u in seq if u != 0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["read", "try", "final", "probe"])
    ap.add_argument("edit", nargs="?")
    ap.add_argument("--accept", action="store_true")
    a = ap.parse_args()
    cfg = load_config()
    out = cfg["out_dir"] / "edit_loop"
    out.mkdir(parents=True, exist_ok=True)
    sp = ROOT / "edits" / "state.json"
    state = json.loads(sp.read_text()) if sp.exists() else {"accepted": [], "passes": 0}
    device = torch.device("cuda")
    audio = Audio(cfg)

    def run(model, units, seg, tag):
        t0 = time.time()
        hyps = decode(model, units, seg, audio, device)
        sc = score(seg, hyps)
        s = summary(sc)
        state["passes"] += 1
        with open(out / "log.jsonl", "a") as f:
            f.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "tag": tag, "half": seg["half"].iloc[0],
                                "stack": state["accepted"], **s}) + "\n")
        print(line(tag, s) + f"  ({time.time() - t0:.0f} s)", flush=True)
        return hyps, sc, s

    if a.cmd == "final":
        seg = module3(cfg, "test")
        model, units = build(cfg, device, [])
        _, b, sb = run(model, units, seg, "test: unedited")
        model, units = build(cfg, device, stack(state))
        _, n, sn = run(model, units, seg, "test: accepted stack")
        lo, hi = gap_ci(b, n)
        print(f"test mean WER change {100 * (sn['macro'] - sb['macro']):+.2f}, gap change {100 * (sn['gap'] - sb['gap']):+.2f} [{100 * lo:+.1f},{100 * hi:+.1f}]")
        state["final"] = {"unedited": sb, "edited": sn, "gap_change_ci": [lo, hi]}
        sp.write_text(json.dumps(state, indent=1))
        return

    seg = module3(cfg, "tune")
    model, units = build(cfg, device, stack(state))
    if a.cmd == "probe":
        probe(model, units, seg, audio, device, a.edit, out)
        return
    hyps, cur, sc = run(model, units, seg, "current stack")
    if a.cmd == "read":
        profile(seg, hyps, cur)
        sp.write_text(json.dumps(state, indent=1))
        return
    path = Path(a.edit).resolve()
    with torch.no_grad():
        load_edit(path).apply(model, units)
    _, new, sn = run(model, units, seg, f"+ {path.name}")
    gain = sc["macro"] - sn["macro"]
    ok = gain >= MIN_GAIN and sn["west"] <= sc["west"]
    lo, hi = gap_ci(cur, new, b=1000)
    print(f"mean WER {100 * -gain:+.2f}, West {100 * (sn['west'] - sc['west']):+.2f}, East {100 * (sn['east'] - sc['east']):+.2f}, "
          f"gap change {100 * (sn['gap'] - sc['gap']):+.2f} [{100 * lo:+.1f},{100 * hi:+.1f}] -> {'passes' if ok else 'fails'} the rule")
    if a.accept and ok:
        state["accepted"].append(str(path.relative_to(ROOT)))
        print(f"accepted ({len(state['accepted'])} edits)")
    state.setdefault("tried", []).append({"edit": path.name, "gain": gain, "west": sn["west"] - sc["west"],
                                          "gap": sn["gap"] - sc["gap"], "accepted": bool(a.accept and ok)})
    sp.write_text(json.dumps(state, indent=1))

if __name__ == "__main__":
    main()
