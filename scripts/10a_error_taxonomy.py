import re
import xml.etree.ElementTree as ET
import zipfile
from collections import defaultdict

import jiwer
import numpy as np
import pandas as pd
from norsk_asr.text import normalize

from dialect_probing.config import load_config
from dialect_probing.written_standard import LEXICONS

HAND = {"eg": "jeg", "æ": "jeg", "je": "jeg", "ej": "jeg", "mæ": "meg", "dæ": "deg", "sæ": "seg", "me": "vi",
        "dokker": "dere", "dokk": "dere", "dåkker": "dere", "dykk": "dere", "ho": "hun", "hu": "hun", "dei": "de",
        "ikkje": "ikke", "ikkj": "ikke", "itj": "ikke", "ittj": "ikke", "ittje": "ikke", "ente": "ikke", "inte": "ikke",
        "kje": "ikke", "ka": "hva", "kva": "hva", "kor": "hvor", "koss": "hvordan", "kossen": "hvordan",
        "korleis": "hvordan", "åssen": "hvordan", "kvifor": "hvorfor", "kven": "hvem", "kem": "hvem", "kjem": "kommer",
        "kjeme": "kommer", "sjå": "se", "vart": "ble", "vore": "vært", "vorre": "vært", "vøri": "vært", "heim": "hjem",
        "berre": "bare", "bærre": "bare", "noko": "noe", "nokon": "noen", "mykje": "mye", "mykkje": "mye",
        "meir": "mer", "viss": "hvis", "òg": "også", "au": "også"}
FUNC_POS = {"pron", "prep", "konj", "sbu", "det", "inf-merke", "interj"}
FUNC_EXTRA = {"er", "var", "har", "hadde", "ha", "skal", "skulle", "vil", "ville", "kan", "kunne", "må", "måtte", "blir",
              "bli", "ble", "blitt", "vært", "være", "så", "da", "jo", "nå", "liksom", "altså", "sånn", "litt", "veldig",
              "også", "bare", "ikke", "der", "her", "nok", "vel", "egentlig", "ganske", "mye", "noe", "hvis"}
NUM = re.compile(r"^(null|en|ett|ei|to|tre|fire|fem|seks|sju|syv|åtte|ni|ti|elleve|tolv|tretten|fjorten|femten|seksten|"
                 r"sytten|atten|nitten|((tjue|tyve|tretti|tredve|førti|femti|seksti|sytti|åtti|nitti)(en|ett|to|tre|fire|fem|"
                 r"seks|sju|syv|åtte|ni)?)|hundre|tusen|millioner?|milliarder?|prosent|komma|første|andre|tredje)$")
CLASSES = ["a1_dialect_form", "a2_dialect_word", "b_name_number", "c_content", "d_function", "insertion"]
B = 10000

def lexicon(root) -> tuple[set, dict, set]:
    forms, tags = set(), defaultdict(set)
    with open(root / LEXICONS["nob"], encoding="latin-1") as f:
        next(f)
        for line in f:
            c = line.split("\t")
            if len(c) > 3:
                w = c[2].lower()
                forms.add(w)
                tags[w].add(c[3].split()[0])
    nno = set()
    with open(root / LEXICONS["nno"], encoding="latin-1") as f:
        next(f)
        for line in f:
            c = line.split("\t")
            if len(c) > 2:
                nno.add(c[2].lower())
    return forms, tags, nno - forms

def slang_pairs(annotation_zip) -> dict:
    root = ET.fromstring(zipfile.ZipFile(annotation_zip).read("Annotation/part_3.xml"))
    pairs = {}
    for el in root.iter("slang"):
        d, n = normalize(el.get("text") or ""), normalize(el.get("normalized") or "")
        if d and n and len(d.split()) == 1:
            pairs[d] = n
    return pairs

def main():
    cfg = load_config()
    out = cfg["out_dir"] / "spont"
    forms, tags, nno_only = lexicon(cfg["data_dir"] / "ordbank")
    slang = slang_pairs(cfg["data_dir"] / "nbtale" / "raw" / "Annotation.zip")
    dialect_forms = (set(HAND) | set(slang) | nno_only) - forms
    targets = set(HAND.values())
    print(f"dialect forms {len(dialect_forms)} (hand {len(HAND)}, module-3 slang {len(slang)}, Nynorsk-only {len(nno_only)}); targets {len(targets)}")

    seg = pd.read_parquet(cfg["data_dir"] / "spont" / "segments.parquet")
    seg = seg[(seg["set"] == "nbtale_m3") & seg["area"].isin(["east", "west", "north", "mid", "central", "south"])]
    hyp = pd.read_parquet(out / "hyp_norsk-asr.parquet")
    names = set()
    for raw in seg["ref_raw"]:
        toks = raw.split()
        for i, t in enumerate(toks):
            w = normalize(t)
            if i > 0 and t[:1].isupper() and w and w not in forms:
                names.add(w)

    def classify(r: str | None, h: str | None) -> str:
        if r is None:
            return "insertion"
        if NUM.match(r) or r in names:
            return "b_name_number"
        if h and h in dialect_forms:
            return "a1_dialect_form"
        if r in targets:
            return "a2_dialect_word"
        if r in FUNC_EXTRA or tags.get(r, set()) & FUNC_POS:
            return "d_function"
        return "c_content"

    rows, cases = [], []
    for dec in ("norsk-asr greedy", "norsk-asr lm"):
        d = seg.merge(hyp[hyp["model"] == dec][["seg_id", "hyp"]], on="seg_id")
        for s in d.itertuples():
            ref, h = s.ref, normalize(s.hyp or "") or "<empty>"
            rw, hw = ref.split(), h.split()
            counts = dict.fromkeys(CLASSES, 0)
            for c in jiwer.process_words(ref, h).alignments[0]:
                if c.type == "equal":
                    continue
                rs, hs = rw[c.ref_start_idx:c.ref_end_idx], hw[c.hyp_start_idx:c.hyp_end_idx]
                if c.type == "insert":
                    counts["insertion"] += len(hs)
                    continue
                for i, r in enumerate(rs):
                    hh = hs[i] if c.type == "substitute" and i < len(hs) else None
                    k = classify(r, hh)
                    counts[k] += 1
                    if dec == "norsk-asr greedy":
                        cases.append({"area": s.area, "speaker": s.speaker, "ref": r, "hyp": hh or "-", "class": k,
                                      "context": " ".join(rw[max(0, c.ref_start_idx - 4):c.ref_end_idx + 4]),
                                      "hyp_context": " ".join(hw[max(0, c.hyp_start_idx - 4):c.hyp_end_idx + 4])})
            rows.append({"decoder": dec, "area": s.area, "speaker": s.speaker, "words": len(rw), **counts})
    t = pd.DataFrame(rows)
    cases = pd.DataFrame(cases)
    t.to_parquet(out / "taxonomy_segments.parquet")
    cases.to_parquet(out / "taxonomy_cases.parquet")

    rng = np.random.default_rng(0)
    res = []
    for dec, g in t.groupby("decoder"):
        spk = g.groupby(["area", "speaker"])[["words"] + CLASSES].sum().reset_index()
        w, e = spk[spk["area"] == "west"], spk[spk["area"] == "east"]
        bw = w[["words"] + CLASSES].to_numpy(float)[rng.integers(0, len(w), (B, len(w)))].sum(1)
        be = e[["words"] + CLASSES].to_numpy(float)[rng.integers(0, len(e), (B, len(e)))].sum(1)
        total_obs = (w[CLASSES].to_numpy().sum() / w["words"].sum()) - (e[CLASSES].to_numpy().sum() / e["words"].sum())
        total_b = bw[:, 1:].sum(1) / bw[:, 0] - be[:, 1:].sum(1) / be[:, 0]
        print(f"\n{dec}: West minus East WER gap {100 * total_obs:+.1f} points "
              f"[{100 * np.percentile(total_b, 2.5):+.1f}, {100 * np.percentile(total_b, 97.5):+.1f}]")
        print("  class            East %   West %   gap (points) [95% CI]      share of gap")
        for i, k in enumerate(CLASSES, 1):
            re_, rw_ = e[k].sum() / e["words"].sum(), w[k].sum() / w["words"].sum()
            gb = bw[:, i] / bw[:, 0] - be[:, i] / be[:, 0]
            share_b = gb / total_b
            res.append({"decoder": dec, "class": k, "east": re_, "west": rw_, "gap": rw_ - re_,
                        "gap_lo": np.percentile(gb, 2.5), "gap_hi": np.percentile(gb, 97.5), "share": (rw_ - re_) / total_obs,
                        "share_lo": np.percentile(share_b, 2.5), "share_hi": np.percentile(share_b, 97.5)})
            r = res[-1]
            print(f"  {k:15s} {100 * re_:6.1f}   {100 * rw_:6.1f}   {100 * r['gap']:+5.1f} [{100 * r['gap_lo']:+.1f}, {100 * r['gap_hi']:+.1f}]"
                  f"     {r['share']:5.0%} [{r['share_lo']:.0%}, {r['share_hi']:.0%}]")
        print("  all areas, % of reference words:")
        a = g.groupby("area")[["words"] + CLASSES].sum()
        print((a[CLASSES].div(a["words"], axis=0) * 100).round(1).to_string())
    pd.DataFrame(res).to_csv(out / "taxonomy_gap.csv", index=False)

    check = cases.sample(50, random_state=50)
    check.to_csv(out / "taxonomy_check50.csv", index=False)
    print("\n50 random greedy errors for hand checking:")
    for r in check.itertuples():
        print(f"  [{r[5][:13]:13s}] {r.ref:>14s} -> {r.hyp:14s} | {r.context} || {r.hyp_context}")
    print("\nmost frequent dialect-class pairs, West:")
    print(cases[cases["class"].str.startswith("a") & (cases["area"] == "west")].groupby(["ref", "hyp"]).size().sort_values(ascending=False).head(15).to_string())

if __name__ == "__main__":
    main()
