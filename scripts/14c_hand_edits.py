import time

import numpy as np
import pandas as pd
import torch

from dialect_probing.config import ROOT, load_config
from dialect_probing.weight_space import Audio, decode, gap_ci, load_model, module3, score, summary

def main():
    cfg = load_config()
    device = torch.device("cuda")
    model, units, _, _ = load_model(cfg, device)
    audio = Audio(cfg)
    seg = module3(cfg, "tune")
    proj, bias0, mix0 = model.proj[0], model.head_bias.detach().clone(), model.layer_weights.detach().clone()
    w0 = proj.weight.detach().clone()

    # speaker means of the reader input (layer-mix output), tune half only
    grab = {}
    hook = proj.register_forward_hook(lambda m, i, o: grab.__setitem__("x", i[0].detach()))
    sums, n, area = {}, {}, {}
    with torch.no_grad():
        for r in seg.itertuples():
            w = torch.from_numpy(audio(r.seg_id))[None].to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, ol = model(w, torch.tensor([w.shape[1]], device=device))
            x = grab["x"][0, :ol[0]].float()
            sums[r.speaker], n[r.speaker], area[r.speaker] = sums.get(r.speaker, 0) + x.sum(0), n.get(r.speaker, 0) + len(x), r.area
    hook.remove()
    spk = sorted(sums)
    means, areas = torch.stack([sums[s] / n[s] for s in spk]), np.array([area[s] for s in spk])

    def directions(labels, k):
        d = means[labels == "west"].mean(0) - means[labels == "east"].mean(0)
        mu = torch.stack([means[labels == a].mean(0) for a in np.unique(labels)])
        _, _, vh = torch.linalg.svd(mu - mu.mean(0), full_matrices=False)
        q, _ = torch.linalg.qr(torch.stack([d / d.norm(), *vh[:k]]).T)
        return q[:, :k]

    def reset():
        with torch.no_grad():
            proj.weight.copy_(w0)
            model.head_bias.copy_(bias0)
            model.layer_weights.copy_(mix0)

    rows, base = [], None

    def run(edit, detail=""):
        nonlocal base
        t0 = time.time()
        sc = score(seg, decode(model, units, seg, audio, device))
        base = sc if base is None else base
        s = summary(sc)
        t = sc.groupby("area")[["words", "sub", "del", "ins"]].sum()
        lo, hi = gap_ci(base, sc, b=1000) if sc is not base else (np.nan, np.nan)
        rows.append({"edit": edit, "detail": detail, **s, "gap_change_lo": lo, "gap_change_hi": hi,
                     **{f"{a}_{c}": t.loc[a, c] / t.loc[a, "words"] for a in ("west", "east") for c in ("sub", "del", "ins")}})
        print(f"{edit:24s} {detail:14s} all {100 * s['all']:5.1f} West {100 * s['west']:5.1f} East {100 * s['east']:5.1f} "
              f"gap {100 * s['gap']:+5.1f} change [{100 * lo:+.1f},{100 * hi:+.1f}] ({time.time() - t0:.0f} s)", flush=True)
        reset()

    run("baseline")
    rng = np.random.default_rng(0)
    for k in (1, 3, 5):
        for name, labels in [("erase West-East", areas)] + [(f"erase shuffled {j}", rng.permutation(areas)) for j in range(2)]:
            q = directions(labels, k)
            with torch.no_grad():
                proj.weight.copy_(w0 - (w0 @ q) @ q.T)
            run(name, f"k={k}")
    j = units.pieces.index("▁j")
    for b in (1.0, 2.0):
        with torch.no_grad():
            model.head_bias[j] += b
        run("bias ▁j", f"+{b:g}")
    p0 = torch.softmax(mix0, 0)
    for move in (0.25, 0.5):
        p = p0.clone()
        m = p[12] * move
        p[12] -= m
        p[5:7] += m / 2
        with torch.no_grad():
            model.layer_weights.copy_(torch.log(p))
        run("mix L12 to L5-6", f"{move:.0%}")
    pd.DataFrame(rows).to_csv(ROOT / "results" / "hand_edits.csv", index=False)

if __name__ == "__main__":
    main()
