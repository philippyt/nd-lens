import json
import random

import pandas as pd

from dialect_probing.config import load_config

N = 30
FIELDS = ["problem", "fakta", "standpunkt", "tiltak"]
HELP = ("Etiketter per felt: korrekt / utelatt / feilhort / tillegg / feilattribusjon / nyansetap. "
        "Retning: for / mot / spor / blandet / ikke_nevnt. Kildeteksten er fasit; vurder bare innhold.")

def main():
    cfg = load_config()
    out = cfg["out_dir"] / "npsc"
    sample = pd.read_parquet(out / "sample.parquet").set_index("speech_id")
    notes = {(r["speech_id"], r["run"]): r["note"] for r in map(json.loads, open(out / "notes.jsonl", encoding="utf-8"))}
    ids = sorted({sid for sid, run in notes if run == "asr" and (sid, "ref1") in notes})
    pick = random.Random(30).sample(ids, N)
    rows, key = [], []
    for case, sid in enumerate(pick, 1):
        order = ["ref1", "asr"]
        random.Random(f"{sid}-ref1-asr").shuffle(order)
        row = {"case": case, "kildetekst": sample.loc[sid, "reference"]}
        for tag, run in zip("AB", order):
            for f in FIELDS:
                v = notes[(sid, run)].get(f, "")
                row[f"{tag}_{f}"] = "; ".join(v) if isinstance(v, list) else v
        for tag in "AB":
            for f in FIELDS + ["retning"]:
                row[f"{tag}_{f}_etikett"] = ""
        rows.append(row)
        key.append({"case": case, "speech_id": sid, "A": order[0], "B": order[1]})
    sheet = pd.DataFrame(rows)
    path = out / "manual_sheet.xlsx"
    with pd.ExcelWriter(path) as w:
        sheet.to_excel(w, sheet_name="vurdering", index=False)
        pd.DataFrame({"veiledning": [HELP]}).to_excel(w, sheet_name="veiledning", index=False)
    pd.DataFrame(key).to_csv(out / "manual_key.csv", index=False)
    print("wrote", path, "and the private key manual_key.csv")

if __name__ == "__main__":
    main()
