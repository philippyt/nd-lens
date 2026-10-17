import json
import re
import tarfile
import time
import unicodedata
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

REPO = "NbAiLab/NPSC"
API = "https://data.stortinget.no/eksport"
PERIODS = ["2005-2009", "2009-2013", "2013-2017", "2017-2021", "2021-2025"]
AREAS = {"Østlandsk": "east", "Midlandsk": "east", "Sørlandsk": "south", "Sørvestlandsk": "west",
         "Nordvestlandsk": "west", "Uttrøndersk": "mid", "Østtrøndsk": "mid", "Namdalsk": "mid",
         "Helgelandsk": "north", "Nordlandsk": "north", "Troms-Finnmarks-mål": "north"}
REPLIKK_ON = re.compile(r"replikkordskifte(?!t er)", re.I)
REPLIKK_OFF = re.compile(r"replikkordskiftet er (omme|over|avsluttet)", re.I)
PROCEDURAL = re.compile(r"(har ordet|neste taler|replikkordskifte|presidenten|møtet er|lovlig satt)", re.I)

def days(split: str) -> list[str]:
    files = HfApi().list_repo_files(REPO, repo_type="dataset")
    return sorted({Path(f).name.split(".")[0] for f in files if re.match(rf"data/{split}/\d+\.json$", f)})

def fetch_split(root: Path, split: str) -> pd.DataFrame:
    rows = []
    for day in days(split):
        meta = hf_hub_download(REPO, f"data/{split}/{day}.json", repo_type="dataset")
        tgz = Path(hf_hub_download(REPO, f"data/{split}/{day}_16K_mp3.tar.gz", repo_type="dataset"))
        out = root / "audio" / split / day
        if not (out / ".done").exists():
            out.mkdir(parents=True, exist_ok=True)
            with tarfile.open(tgz) as t:
                t.extractall(out, filter="data")
            (out / ".done").touch()
        files = {p.stem: p for p in out.rglob("*.mp3")}
        for line in open(meta, encoding="utf-8"):
            r = json.loads(line)
            r["audio"] = str(files.get(Path(r["path"]).name, ""))
            rows.append(r)
    return pd.DataFrame(rows)

def _get(url: str, headers: dict | None = None) -> dict:
    for i in range(6):
        time.sleep(0.4)
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers or {}), timeout=30) as f:
                return json.load(f)
        except Exception:
            time.sleep(2 ** i)
    raise RuntimeError(f"failed: {url}")

def wikidata_birthplace(name: str) -> list[str]:
    q = ("SELECT ?pl ?mu WHERE { ?p rdfs:label \"%s\"@nb; wdt:P27 wd:Q20; wdt:P19 ?b. "
         "?b rdfs:label ?pl FILTER(lang(?pl)='nb') "
         "OPTIONAL { ?b wdt:P131* ?m. ?m wdt:P31 wd:Q755707; rdfs:label ?mu FILTER(lang(?mu)='nb') } }") % name.replace('"', "")
    url = "https://query.wikidata.org/sparql?format=json&query=" + urllib.parse.quote(q)
    try:
        res = _get(url, {"User-Agent": "nd-lens research (philipaha@proton.me)"})["results"]["bindings"]
    except RuntimeError:
        return []
    return [v["value"] for r in res for k in ("mu", "pl") if (v := r.get(k))]

def _key(name: str) -> str:
    s = unicodedata.normalize("NFKC", name).lower().replace("-", " ")
    return " ".join(s.split())

def people() -> pd.DataFrame:
    rows = []
    for p in PERIODS:
        for r in _get(f"{API}/representanter?stortingsperiodeid={p}&format=json").get("representanter_liste", []):
            rows.append({"person_id": r["id"], "first": r["fornavn"], "last": r["etternavn"], "period": p,
                         "party": (r.get("parti") or {}).get("id"), "rep_county": (r.get("fylke") or {}).get("navn"),
                         "deputy": r.get("vara_representant", False)})
    return pd.DataFrame(rows)

def match_speakers(names: pd.Series, ppl: pd.DataFrame) -> pd.DataFrame:
    full = {_key(f"{r.first} {r.last}"): r.person_id for r in ppl.itertuples()}
    short = {}
    for r in ppl.itertuples():
        short.setdefault(_key(f"{r.first.split()[0]} {r.last}"), set()).add(r.person_id)
        short.setdefault(_key(f"{r.first.split()[0]} {r.last.split()[-1]}"), set()).add(r.person_id)
    out = []
    for n in names:
        k = _key(n)
        pid = full.get(k)
        if pid is None:
            toks = k.split()
            cand = short.get(f"{toks[0]} {' '.join(toks[1:])}", set()) | short.get(f"{toks[0]} {toks[-1]}", set())
            pid = next(iter(cand)) if len(cand) == 1 else None
        out.append({"speaker_name": n, "person_id": pid})
    return pd.DataFrame(out)

def birthplaces(ids: list[str]) -> pd.DataFrame:
    rows = []
    for pid in ids:
        p = _get(f"{API}/kodetbiografi?personid={urllib.parse.quote(pid)}&format=json").get("personalia_kodet") or {}
        rows.append({"person_id": pid, "birth_muni": p.get("foede_kommune"), "birth_county": p.get("foede_fylke")})
    return pd.DataFrame(rows)

def dialect_table(spk: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    muni, county = {}, {}
    for col in ("new_muni", "old_muni"):
        for k, g in mapping.groupby(mapping[col].str.lower().str.strip()):
            muni.setdefault(k, g["named_dialect"].mode().iloc[0])
    for col in ("new_county", "old_county", "new_county_2024"):
        for k, g in mapping.groupby(mapping[col].str.lower().str.strip()):
            county.setdefault(k, g["named_dialect"].mode().iloc[0])
    low = lambda x: str(x).lower().strip() if isinstance(x, str) else ""
    rows = []
    for r in spk.itertuples():
        chain = [("birth_muni", muni.get(low(r.birth_muni)))]
        chain += [("wikidata", muni.get(low(p))) for p in (r.wikidata or [])]
        chain += [("birth_county", county.get(low(r.birth_county))), ("rep_county", county.get(low(r.rep_county)))]
        src, named = next(((s, d) for s, d in chain if d), (None, None))
        rows.append({"speaker_name": r.speaker_name, "named_dialect": named, "area": AREAS.get(named), "dialect_source": src})
    return spk.merge(pd.DataFrame(rows), on="speaker_name")

def speeches(sent: pd.DataFrame) -> pd.DataFrame:
    sent = sent.sort_values(["meeting_date", "sentence_order"])
    sid = (sent["speaker_id"] != sent.groupby("meeting_date")["speaker_id"].shift()).cumsum()
    sent = sent.assign(speech_id=sid)
    sp = sent.groupby("speech_id").agg(meeting_date=("meeting_date", "first"), data_split=("data_split", "first"),
                                       speaker_name=("speaker_name", "first"), start=("start_time", "min"),
                                       end=("end_time", "max"), sentences=("sentence_id", "size"),
                                       ref_text=("ref", " ".join), nn_sentences=("is_nn", "sum")).reset_index()
    sp["seconds"] = (sp["end"] - sp["start"]) / 1000
    kind, state = [], {}
    for r in sp.itertuples():
        on = state.get(r.meeting_date, False)
        procedural = bool(PROCEDURAL.search(r.ref_text)) and r.seconds < 40
        kind.append("procedural" if procedural else ("replikk" if on and r.seconds <= 150 else "main"))
        if REPLIKK_OFF.search(r.ref_text):
            state[r.meeting_date] = False
        elif REPLIKK_ON.search(r.ref_text):
            state[r.meeting_date] = True
    sp["kind"] = kind
    return sp, sent
