import re

import numpy as np
import pandas as pd
from norsk_asr.text import normalize

from dialect_probing.config import load_config
from dialect_probing.npsc import (birthplaces, dialect_table, fetch_split, match_speakers, people, speeches,
                                  wikidata_birthplace)

def main():
    cfg = load_config()
    root = cfg["data_dir"] / "npsc"
    root.mkdir(exist_ok=True)
    sent = pd.concat([fetch_split(root, s) for s in ("test", "eval")], ignore_index=True)
    sent["is_nn"] = sent["sentence_language_code"] == "nn-NO"
    raw = np.where(sent["is_nn"], sent["transsentence_text"], sent["text"]).astype(str)
    sent["ref_raw"] = raw
    sent["ref"] = [normalize(re.sub(r"<[^>]*>", " ", t)) for t in raw]
    sent = sent[(sent["ref"].str.len() > 0) & (sent["audio"] != "")]
    print(f"sentences {len(sent)} ({sent['is_nn'].sum()} Nynorsk-transcribed, Bokmål reference from NPSC's translation)")
    print("hours by split:", (sent.groupby("data_split").apply(lambda g: (g["end_time"] - g["start_time"]).sum()) / 3.6e6).round(2).to_dict())

    ppl = people()
    names = pd.Series(sorted(sent["speaker_name"].dropna().unique()))
    m = match_speakers(names, ppl)
    print(f"speakers {len(names)}, matched to Stortinget persons {m['person_id'].notna().sum()}")
    print("unmatched:", m.loc[m["person_id"].isna(), "speaker_name"].tolist())
    births = birthplaces(m["person_id"].dropna().unique().tolist())
    order = {p: i for i, p in enumerate(["2017-2021", "2013-2017", "2021-2025", "2009-2013", "2005-2009"])}
    party = (ppl.assign(o=ppl["period"].map(order)).sort_values("o").groupby("person_id")[["party", "rep_county"]]
             .first().reset_index())
    spk = m.merge(births, on="person_id", how="left").merge(party, on="person_id", how="left")
    need = spk["birth_muni"].isna() & (spk["speaker_name"] != "unknown")
    spk["wikidata"] = None
    spk.loc[need, "wikidata"] = spk.loc[need, "speaker_name"].map(wikidata_birthplace)
    spk = dialect_table(spk, pd.read_csv(cfg["data_dir"] / "dialektmapping" / "dialect_mapping.csv"))
    print("dialect source:", spk["dialect_source"].value_counts(dropna=False).to_dict())
    print("no area:", spk.loc[spk["area"].isna(), "speaker_name"].tolist())

    sp, sent = speeches(sent)
    spk = spk.drop(columns=["wikidata"])
    sp = sp.merge(spk, on="speaker_name", how="left")
    sent.to_parquet(root / "sentences.parquet")
    sp.to_parquet(root / "speeches.parquet")
    spk.to_parquet(root / "speakers.parquet")

    sub = sp[sp["kind"] != "procedural"]
    print("\nspeeches by kind:", sp["kind"].value_counts().to_dict())
    t = sub.groupby(["area", "kind"]).agg(speakers=("speaker_name", "nunique"), speeches=("speech_id", "size"),
                                          minutes=("seconds", lambda s: s.sum() / 60)).round(1)
    print(t.unstack("kind").to_string())
    print("\nper area (non-procedural): speakers, minutes, parties")
    for a, g in sub.groupby(sp["area"].fillna("unknown")):
        print(f"  {a:8s} speakers {g['speaker_name'].nunique():3d}  minutes {g['seconds'].sum() / 60:6.1f}  "
              f"parties {g.drop_duplicates('speaker_name')['party'].value_counts().to_dict()}")

if __name__ == "__main__":
    main()
