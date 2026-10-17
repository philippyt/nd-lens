import bisect
import json
import re
import time
import unicodedata
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta

import jiwer
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from huggingface_hub import hf_hub_download
from norsk_asr.text import has_digits, is_clean, normalize
from scipy.stats import spearmanr

from dialect_probing.config import load_config
from dialect_probing.npsc import days

API = "https://data.stortinget.no/eksport"
NAVN = re.compile(r"^(?:(?:Statsråd|Statsminister|[A-ZÆØÅ][\w-]*minister|Stortingspresident|Presidenten)\s+)?(.+?)\s*(?:\([^)]*\))?\s*\[(\d{2}):(\d{2}):(\d{2})\]")

def get(url: str) -> bytes:
    for i in range(6):
        time.sleep(0.3)
        try:
            with urllib.request.urlopen(url, timeout=60) as f:
                return f.read()
        except Exception:
            time.sleep(2 ** i)
    return b""

def key(name: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", name).lower().replace("-", " ").split())

def referat_ids(cache) -> pd.DataFrame:
    path = cache / "referat_ids.parquet"
    if path.exists():
        return pd.read_parquet(path)
    rows = []
    for y in range(2009, 2024):
        s = f"{y}-{y + 1}"
        d = json.loads(get(f"{API}/publikasjoner?publikasjontype=referat&sesjonid={s}&format=json") or b"{}")
        for r in d.get("publikasjoner_liste", []):
            if r["id"].startswith(("s", "refs-")):
                ms = int(re.search(r"\((-?\d+)", r["dato"]).group(1))
                rows.append({"ref_id": r["id"], "date": (datetime(1970, 1, 1) + timedelta(milliseconds=ms + 3 * 3600 * 1000)).strftime("%Y%m%d")})
    df = pd.DataFrame(rows)
    df.to_parquet(path)
    return df

def turns(ref_id: str, date: str, cache) -> list[dict]:
    path = cache / f"{ref_id}.xml"
    if not path.exists():
        path.write_bytes(get(f"{API}/publikasjon?publikasjonid={ref_id}"))
    try:
        root = ET.fromstring(path.read_bytes())
    except ET.ParseError:
        return []
    out = []
    for inn in root.iter():
        if inn.tag.lower() not in ("innlegg", "hovedinnlegg", "replikk", "presinnlegg"):
            continue
        n = next((e for e in inn.iter() if e.tag.lower() == "navn"), None)
        m = NAVN.match(" ".join("".join(n.itertext()).split())) if n is not None else None
        if not m:
            continue
        t = datetime.strptime(date, "%Y%m%d") + timedelta(hours=int(m.group(2)), minutes=int(m.group(3)), seconds=int(m.group(4)))
        text = normalize(" ".join("".join(a.itertext()) for a in inn.iter() if a.tag.lower() == "a"))
        out.append({"t": t, "speaker": m.group(1), "text": text})
    return out

def seg_time(seg_id: str) -> datetime | None:
    m = re.match(r"stortinget-(\d{8})-(\d{6})(?:_\w+?)?_(\d+)_(\d+)$", seg_id)
    if not m:
        return None
    start = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
    return start + timedelta(milliseconds=(int(m.group(3)) + int(m.group(4))) / 2)

def main():
    cfg = load_config()
    out = cfg["out_dir"] / "npsc"
    cache = cfg["data_dir"] / "referat"
    cache.mkdir(exist_ok=True)
    idx = pd.concat([pd.read_parquet(p) for p in sorted((cfg["data_dir"] / "train_index").glob("*.parquet"))])
    idx = idx[idx["kept"]].copy()
    idx["date"] = idx["id"].str.extract(r"stortinget-(\d{8})")[0]
    npsc_dates = set(pd.read_parquet(cfg["data_dir"] / "npsc" / "sentences.parquet")["meeting_date"].astype(str))
    print(f"kept training segments {len(idx)} ({idx['seconds'].sum() / 3600:.0f} h); on NPSC test/eval dates: {idx['date'].isin(npsc_dates).sum()}")

    ids = referat_ids(cache)
    by_date = ids.groupby("date")["ref_id"].apply(list).to_dict()
    assigned = []
    for date, g in idx.groupby("date"):
        tl = sorted((x for r in by_date.get(date, []) for x in turns(r, date, cache)), key=lambda x: x["t"])
        if not tl:
            continue
        times = [x["t"] for x in tl]
        for s in g.itertuples():
            t = seg_time(s.id)
            if t is None:
                continue
            i = bisect.bisect_right(times, t) - 1
            words = normalize(s.text).split()
            probe = " ".join(words[2:8]) if len(words) >= 8 else " ".join(words)
            match = None
            for j in (i, i - 1, i + 1):
                if 0 <= j < len(tl) and probe and probe in tl[j]["text"]:
                    match = j
                    break
            assigned.append({"id": s.id, "date": date, "seconds": s.seconds, "time_speaker": tl[i]["speaker"] if i >= 0 else None,
                             "speaker": tl[match]["speaker"] if match is not None else None, "text_verified": match is not None})
    a = pd.DataFrame(assigned)
    a.to_parquet(out / "train_speakers.parquet")
    print(f"segments with a referat for their date: {len(a)} of {len(idx)}; text-verified speaker {a['text_verified'].mean():.1%}; "
          f"verified agrees with time-based {(a.loc[a['text_verified'], 'speaker'] == a.loc[a['text_verified'], 'time_speaker']).mean():.1%}")

    npsc_train = []
    for day in days("train"):
        for line in open(hf_hub_download("NbAiLab/NPSC", f"data/train/{day}.json", repo_type="dataset"), encoding="utf-8"):
            r = json.loads(line)
            if r["sentence_language_code"] != "nb-NO":
                continue
            orig = r["text"] or ""
            text = normalize(re.sub(r"<[^>]*>", " ", orig))
            if has_digits(orig) or not is_clean(text):
                continue
            npsc_train.append({"speaker": r["speaker_name"], "seconds": (r["end_time"] - r["start_time"]) / 1000})
    nt = pd.DataFrame(npsc_train)

    sent = pd.read_parquet(cfg["data_dir"] / "npsc" / "sentences.parquet").merge(pd.read_parquet(out / "hyp_norsk-asr.parquet")[["sentence_id", "greedy", "lm"]], on="sentence_id")
    spk = pd.read_parquet(cfg["data_dir"] / "npsc" / "speakers.parquet")[["speaker_name", "area", "party"]]
    sent = sent[~sent["is_nn"] & ~sent["ref_raw"].str.contains(r"\d") & (sent["speaker_name"] != "unknown")]
    rows = []
    for name, g in sent.groupby("speaker_name"):
        r = {"speaker_name": name, "test_minutes": (g["end_time"] - g["start_time"]).sum() / 60000, "sentences": len(g)}
        for dec in ("greedy", "lm"):
            o = jiwer.process_words(list(g["ref"]), [h or "<empty>" for h in g[dec]])
            r[f"wer_{dec}"] = (o.substitutions + o.deletions + o.insertions) / (o.hits + o.substitutions + o.deletions)
        rows.append(r)
    w = pd.DataFrame(rows).merge(spk, on="speaker_name", how="left")
    st_hours = a.dropna(subset=["speaker"]).assign(k=lambda x: x["speaker"].map(key)).groupby("k")["seconds"].sum() / 3600
    nt_hours = nt.assign(k=nt["speaker"].map(key)).groupby("k")["seconds"].sum() / 3600
    w["k"] = w["speaker_name"].map(key)
    w["stortinget_h"] = w["k"].map(st_hours).fillna(0)
    w["npsc_train_h"] = w["k"].map(nt_hours).fillna(0)
    w["train_h"] = w["stortinget_h"] + w["npsc_train_h"]
    w.drop(columns=["k"]).to_csv(out / "speaker_exposure.csv", index=False)

    print(f"\nNPSC test/eval speakers {len(w)}: in Stortinget training {(w['stortinget_h'] > 0).sum()}, "
          f"in NPSC train {(w['npsc_train_h'] > 0).sum()}, in either {(w['train_h'] > 0).sum()}")
    print("training hours per speaker: median {:.2f}, IQR {:.2f}-{:.2f}, max {:.1f}".format(
        w["train_h"].median(), w["train_h"].quantile(.25), w["train_h"].quantile(.75), w["train_h"].max()))
    print(w.groupby("area")[["train_h", "wer_greedy"]].median().round(3).to_string())
    m = w[w["sentences"] >= 20]
    for dec in ("greedy", "lm"):
        rho, p = spearmanr(np.log1p(m["train_h"]), m[f"wer_{dec}"])
        print(f"\nWER ({dec}) vs log(1 + training hours), speakers with >= 20 sentences (n={len(m)}): Spearman {rho:+.2f} (p {p:.3f})")
        for lab, g in m.groupby(pd.cut(m["train_h"], [-0.01, 0, 1, 5, 1e9], labels=["0 h", "0-1 h", "1-5 h", "> 5 h"]), observed=True):
            print(f"  {lab:6s} speakers {len(g):3d}  median WER {100 * g[f'wer_{dec}'].median():5.1f}")
    fit = smf.wls("wer_greedy ~ np.log1p(train_h) + C(area, Treatment('east'))", data=m, weights=m["sentences"]).fit()
    print("\nweighted regression, WER greedy ~ log(1 + training h) + area:")
    print(fit.summary2().tables[1][["Coef.", "[0.025", "0.975]"]].round(3).to_string())

if __name__ == "__main__":
    main()
