import json

import numpy as np
import pandas as pd
from norsk_asr.lm import ctc_labels, make_decoder
from norsk_asr.units import Units

from dialect_probing.config import load_config
from dialect_probing.m3_decode import expand, gap_scores, greedy_words, lm_decode, load_logprobs, word_confidence

BUDGET = 0.20
RETUNED = (0.4, 2.0, 1.0)
PILOT = {"west": 10, "east": 10, "other": 10}

def main():
    cfg = load_config()
    out = cfg["out_dir"] / "spont"
    llm = out / "llm"
    llm.mkdir(exist_ok=True)
    units = Units.load(cfg["checkpoint"] / "units.json")
    ix = pd.read_parquet(out / "m3_logprobs_index.parquet").merge(pd.read_csv(out / "m3_split.csv")[["speaker", "half"]], on="speaker")
    seg = pd.read_parquet(cfg["data_dir"] / "spont" / "segments.parquet").set_index("seg_id")
    ix["start"] = ix["seg_id"].map(seg["start"])
    lps = load_logprobs(out / "m3_logprobs.npz")
    dec = make_decoder(str(cfg["norsk_asr_dir"] / "data" / "lm" / "no4"), ctc_labels(units), RETUNED[0], RETUNED[1])
    hyps = {"greedy": [greedy_words(lp, units.pieces) for lp in lps],
            "lm_retuned": lm_decode(dec, lps, list(range(len(lps))), RETUNED[2])}
    items = []
    tune = ix["half"].eq("tune").to_numpy()
    for name, words in hyps.items():
        per = []
        for i, w in enumerate(words):
            toks, spans = expand(w)
            conf = np.array([word_confidence(lps[i], sp)[1] for sp in spans])
            gaps, _ = gap_scores(lps[i], spans)
            per.append((toks, conf, gaps))
        wthr = np.quantile(np.concatenate([p[1] for p, t in zip(per, tune) if t and len(p[1])]), BUDGET)
        gthr = np.quantile(np.concatenate([p[2] for p, t in zip(per, tune) if t]), 1 - BUDGET)
        print(f"{name}: thresholds from tune half, word conf_min <= {wthr:.3f}, gap mass >= {gthr:.3f}")
        prev = {}
        for i in ix.sort_values(["speaker", "start"]).index:
            r = ix.loc[i]
            toks, conf, gaps = per[i]
            if r["half"] == "test":
                fw, fg = (conf <= wthr).tolist(), (gaps >= gthr).tolist()
                parts, slots = [], []
                for k in range(len(toks) + 1):
                    if fg[k]:
                        parts.append("[_]")
                        slots.append({"kind": "gap", "pos": k})
                    if k < len(toks):
                        if fw[k]:
                            parts.append(f"[{toks[k]}]")
                            slots.append({"kind": "word", "pos": k})
                        else:
                            parts.append(toks[k])
                items.append({"id": f"{name}|{i}", "decoder": name, "seg": int(i), "speaker": r["speaker"], "area": r["area"],
                              "text": " ".join(toks), "marked": " ".join(parts), "slots": slots, "context": prev.get(r["speaker"], "")})
            prev[r["speaker"]] = " ".join(toks)
    with open(llm / "items.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in items)
    test = ix[ix["half"] == "test"].copy()
    test["grp"] = test["area"].where(test["area"].isin(["west", "east"]), "other")
    rng = np.random.default_rng(0)
    pilot = sorted(int(s) for g, n in PILOT.items() for s in rng.choice(test.index[test["grp"] == g], n, replace=False))
    (llm / "pilot_segs.json").write_text(json.dumps(pilot))
    print(f"{len(items)} items ({len(test)} test segments x {len(hyps)} decoders), {sum(bool(x['slots']) for x in items)} with flags; "
          f"mean slots per item {np.mean([len(x['slots']) for x in items]):.1f}; pilot segments {len(pilot)}")

if __name__ == "__main__":
    main()
