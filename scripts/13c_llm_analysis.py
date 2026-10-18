import argparse
import difflib
import json

import jiwer
import numpy as np
import pandas as pd
from norsk_asr.text import normalize

from dialect_probing.config import ROOT, load_config
from dialect_probing.m3_decode import B
from dialect_probing.variants import load_lexicon, same_variant

MIN_RATIO, MAX_RATIO = 0.5, 1.5
SHORT, SHORT_MAX_ADD = 5, 3
AREAS = ["all", "west", "east", "central", "north", "south", "mixed"]
RATES = ["broken", "fixed", "invented", "new_wrong"]

def reconstruct(item: dict, variant: str, res: dict | None) -> tuple[list[str], str, int]:
    toks = item["text"].split()
    if not res:
        return toks, "no response", 0
    out = normalize(res.get("tekst", "").replace("[", " ").replace("]", " ").replace("_", " ")).split()
    n = len(toks)
    if not MIN_RATIO * n <= len(out) <= MAX_RATIO * n + 3 or (n <= SHORT and len(out) > n + SHORT_MAX_ADD):
        return toks, "length guard", 0
    if variant == "a":
        return out, "", 0
    words = {s["pos"] for s in item["slots"] if s["kind"] == "word"}
    gaps = {s["pos"] for s in item["slots"] if s["kind"] == "gap"}
    kept, dropped = [], 0
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=toks, b=out, autojunk=False).get_opcodes():
        ok = tag == "equal" or (tag == "insert" and i1 in gaps) or (tag != "insert" and all(k in words for k in range(i1, i2)))
        kept += out[j1:j2] if ok else toks[i1:i2]
        dropped += not ok
    return kept, "", dropped

def status(ref: list[str], hyp: list[str], forms: dict | None) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    ok, hyp_ok, ins = np.zeros(len(ref), bool), np.zeros(len(hyp), bool), np.zeros(len(hyp), bool)
    if not hyp:
        return ok, hyp_ok, ins, len(ref)
    err = 0
    for a in jiwer.process_words(" ".join(ref), " ".join(hyp)).alignments[0]:
        if a.type == "equal":
            ok[a.ref_start_idx:a.ref_end_idx] = True
            hyp_ok[a.hyp_start_idx:a.hyp_end_idx] = True
        elif a.type == "insert":
            ins[a.hyp_start_idx:a.hyp_end_idx] = True
            err += a.hyp_end_idx - a.hyp_start_idx
        elif a.type == "delete":
            err += a.ref_end_idx - a.ref_start_idx
        else:
            for i, j in zip(range(a.ref_start_idx, a.ref_end_idx), range(a.hyp_start_idx, a.hyp_end_idx)):
                if forms is not None and same_variant(forms, ref[i], hyp[j]):
                    ok[i] = hyp_ok[j] = True
                else:
                    err += 1
    return ok, hyp_ok, ins, err

def compare(ref: str, before: list[str], after: list[str], forms: dict) -> dict:
    r = ref.split()
    new = np.zeros(len(after), bool)
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=before, b=after, autojunk=False).get_opcodes():
        if tag in ("insert", "replace"):
            new[j1:j2] = True
    out = {"words": len(r), "changed": int(before != after), "new_words": int(new.sum())}
    for sfx, fm in (("", None), ("_tol", forms)):
        ok_b, _, _, eb = status(r, before, fm)
        ok_a, hok_a, ins_a, ea = status(r, after, fm)
        out |= {f"err_before{sfx}": eb, f"err_after{sfx}": ea, f"broken{sfx}": int((ok_b & ~ok_a).sum()),
                f"fixed{sfx}": int((~ok_b & ok_a).sum()), f"new_wrong{sfx}": int((new & ~hok_a).sum()),
                f"invented{sfx}": int((new & ins_a).sum())}
    return out

def area_stats(sp: pd.DataFrame, sfx: str, rng) -> dict:
    cols = ["words", f"err_before{sfx}", f"err_after{sfx}"] + [f"{k}{sfx}" for k in RATES]
    a = sp[cols].to_numpy(float)
    s = a[rng.integers(0, len(a), (B, len(a)))].sum(1)
    ci = lambda x: tuple(np.percentile(100 * x, [2.5, 97.5]))
    out = {"wer_before": 100 * a[:, 1].sum() / a[:, 0].sum(), "wer_after": 100 * a[:, 2].sum() / a[:, 0].sum()}
    out["change"] = out["wer_after"] - out["wer_before"]
    out["change_lo"], out["change_hi"] = ci((s[:, 2] - s[:, 1]) / s[:, 0])
    for j, k in enumerate(RATES, start=3):
        out[k] = 100 * a[:, j].sum() / a[:, 0].sum()
        out[f"{k}_lo"], out[f"{k}_hi"] = ci(s[:, j] / s[:, 0])
    return out

def west_east(sp: pd.DataFrame, sfx: str, rng) -> dict:
    res, obs = [], []
    for area in ("west", "east"):
        a = sp[sp["area"] == area][["words", f"err_before{sfx}", f"err_after{sfx}"]].to_numpy(float)
        res.append(a[rng.integers(0, len(a), (B, len(a)))].sum(1))
        obs.append(a.sum(0))
    (w, e), (ow, oe) = res, obs
    gb, ga = w[:, 1] / w[:, 0] - e[:, 1] / e[:, 0], w[:, 2] / w[:, 0] - e[:, 2] / e[:, 0]
    rb, ra = (w[:, 1] / w[:, 0]) / (e[:, 1] / e[:, 0]), (w[:, 2] / w[:, 0]) / (e[:, 2] / e[:, 0])
    g0, g1 = 100 * (ow[1] / ow[0] - oe[1] / oe[0]), 100 * (ow[2] / ow[0] - oe[2] / oe[0])
    r0, r1 = (ow[1] / ow[0]) / (oe[1] / oe[0]), (ow[2] / ow[0]) / (oe[2] / oe[0])
    p = lambda x, m=100: tuple(np.percentile(m * x, [2.5, 97.5]))
    return {"gap_before": g0, "gap_after": g1, "gap_after_lo": p(ga)[0], "gap_after_hi": p(ga)[1], "gap_change": g1 - g0,
            "gap_change_lo": p(ga - gb)[0], "gap_change_hi": p(ga - gb)[1], "ratio_before": r0, "ratio_after": r1,
            "ratio_change_lo": p(ra - rb, 1)[0], "ratio_change_hi": p(ra - rb, 1)[1]}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pilot", action="store_true")
    a = p.parse_args()
    cfg = load_config()
    llm = cfg["out_dir"] / "spont" / "llm"
    forms, _ = load_lexicon(cfg["data_dir"] / "ordbank")
    ix = pd.read_parquet(cfg["out_dir"] / "spont" / "m3_logprobs_index.parquet")
    items = {x["id"]: x for x in map(json.loads, (llm / "items.jsonl").read_text(encoding="utf-8").splitlines())}
    resp = [json.loads(l) for l in (llm / ("pilot.jsonl" if a.pilot else "corrections.jsonl")).read_text(encoding="utf-8").splitlines()]
    got = {(r["id"], r["variant"]): r for r in resp}
    segs = {int(i.split("|")[1]) for i, _ in got}
    print(f"{len(resp)} responses, {sum(r['error'] is not None for r in resp)} errors, "
          f"median {np.median([r['seconds'] for r in resp]):.1f} s per call, model {resp[0]['model']}, prompt {resp[0]['prompt_hash']}")
    rows, examples = [], []
    for iid, item in items.items():
        if item["seg"] not in segs:
            continue
        ref = ix.loc[item["seg"], "ref"]
        before = item["text"].split()
        for variant in ("a", "b"):
            r = got.get((iid, variant))
            called = variant == "a" or bool(item["slots"])
            after, note, dropped = reconstruct(item, variant, r["response"] if r else None) if called else (before, "", 0)
            m = compare(ref, before, after, forms)
            rows.append({"decoder": item["decoder"], "variant": variant, "seg": item["seg"], "speaker": item["speaker"], "area": item["area"],
                         "calls": int(called), "failed": int(called and note == "no response"), "guarded": int(note == "length guard"),
                         "short": int(len(before) <= SHORT), "dropped_edits": dropped, "segs_with_dropped": int(dropped > 0), **m})
            examples.append({"id": iid, "variant": variant, "area": item["area"], "ref": ref, "before": " ".join(before),
                             "after": " ".join(after), "raw": ((r or {}).get("response") or {}).get("tekst", ""),
                             "marked": item["marked"], "note": note, "dropped_edits": dropped, **m})
    d = pd.DataFrame(rows)
    pd.DataFrame(examples).to_parquet(llm / ("pilot_examples.parquet" if a.pilot else "examples.parquet"))
    rng = np.random.default_rng(0)
    out, gaps = [], []
    for (dec, var), g in d.groupby(["decoder", "variant"]):
        sp = g.groupby(["area", "speaker"]).sum(numeric_only=True).reset_index()
        for scoring, sfx in (("variant-tolerant", "_tol"), ("plain", "")):
            for area in AREAS:
                s = sp if area == "all" else sp[sp["area"] == area]
                if len(s):
                    out.append({"decoder": dec, "variant": var, "scoring": scoring, "area": area, "speakers": len(s),
                                "calls": int(s["calls"].sum()), "failed": int(s["failed"].sum()), "guarded": int(s["guarded"].sum()),
                                "segs_with_dropped": int(s["segs_with_dropped"].sum()), "dropped_edits": int(s["dropped_edits"].sum()),
                                **area_stats(s, sfx, rng)})
            if not a.pilot:
                gaps.append({"decoder": dec, "variant": var, "scoring": scoring, **west_east(sp, sfx, rng)})
    res = pd.DataFrame(out)
    pd.set_option("display.width", 250, "display.max_columns", 40)
    cols = ["decoder", "variant", "area", "speakers", "calls", "failed", "guarded", "segs_with_dropped", "dropped_edits", "wer_before", "wer_after",
            "change", "change_lo", "change_hi", "broken", "fixed", "invented", "new_wrong"]
    for scoring in ("variant-tolerant", "plain"):
        print(f"\n{scoring} scoring (rates per 100 reference words; 95% CI over speakers)\n"
              + res[res["scoring"] == scoring][cols].round(1).to_string(index=False))
    if not a.pilot:
        gaps = pd.DataFrame(gaps)
        res.to_csv(ROOT / "results" / "llm_correction.csv", index=False)
        gaps.to_csv(ROOT / "results" / "llm_correction_gap.csv", index=False)
        print("\nWest minus East WER gap before and after LLM correction, paired 95% CI over speakers\n" + gaps.round(2).to_string(index=False))

if __name__ == "__main__":
    main()
