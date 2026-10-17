import argparse
import hashlib
import json
import random
import time
import urllib.request
from pathlib import Path

HOST = "http://127.0.0.1:11434"
MODEL = "qwen3.5:9b"
PAIRS = [("ref1", "asr"), ("ref1", "ref2")]
FIELDS = ["problem", "fakta", "standpunkt", "tiltak"]
LABELS = ["korrekt", "utelatt", "feilhort", "tillegg", "feilattribusjon", "nyansetap"]
DIRECTIONS = ["for", "mot", "spor", "blandet", "ikke_nevnt"]
SYSTEM = (
    "Du kontrollerer notater mot en kildetekst. Kildeteksten er fasit. Vurder hvert felt i notat A og notat B for seg, "
    "bare ut fra om innholdet stemmer med kildeteksten. Ikke vurder politikken, språket eller stilen.\n"
    "Etiketter per felt (velg den alvorligste som gjelder):\n"
    "- korrekt: innholdet stemmer med kildeteksten (også \"Ikke nevnt\" når kilden ikke nevner det)\n"
    "- utelatt: viktig innhold fra kilden mangler i feltet\n"
    "- feilhort: et ord, navn eller tall er feil gjengitt (som om det var hørt feil)\n"
    "- tillegg: feltet påstår noe som ikke står i kilden\n"
    "- feilattribusjon: innhold tilskrives feil person eller sak (for eksempel statsrådens syn presentert som talerens)\n"
    "- nyansetap: innholdet er riktig i hovedsak, men forsterket, svekket eller forenklet\n"
    "Retning for standpunktet: for, mot, spor (taleren spør eller stiller spørsmål ved), blandet, ikke_nevnt."
)
NOTE_SCHEMA = {"type": "object", "properties": {**{f: {"type": "string", "enum": LABELS} for f in FIELDS},
                                                "retning": {"type": "string", "enum": DIRECTIONS}},
               "required": FIELDS + ["retning"]}
SCHEMA = {"type": "object", "properties": {"A": NOTE_SCHEMA, "B": NOTE_SCHEMA}, "required": ["A", "B"]}

def show(note: dict) -> str:
    lst = lambda v: "; ".join(v) if isinstance(v, list) else str(v)
    return "\n".join(f"{k}: {lst(note.get(k, ''))}" for k in FIELDS)

def judge(reference: str, a: dict, b: dict, model: str) -> dict:
    user = f"Kildetekst:\n{reference}\n\nNotat A:\n{show(a)}\n\nNotat B:\n{show(b)}"
    body = {"model": model, "stream": False, "think": False, "format": SCHEMA,
            "options": {"temperature": 0, "seed": 0, "num_ctx": 8192},
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]}
    req = urllib.request.Request(f"{HOST}/api/chat", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as f:
        return json.loads(json.load(f)["message"]["content"])

def main():
    p = argparse.ArgumentParser()
    p.add_argument("sample")
    p.add_argument("notes")
    p.add_argument("out")
    p.add_argument("--model", default=MODEL)
    a = p.parse_args()
    print("judge prompt hash", hashlib.sha256((SYSTEM + json.dumps(SCHEMA)).encode()).hexdigest()[:16], flush=True)
    ref = {r["speech_id"]: r["reference"] for r in map(json.loads, Path(a.sample).read_text(encoding="utf-8").splitlines())}
    notes = {}
    for r in map(json.loads, Path(a.notes).read_text(encoding="utf-8").splitlines()):
        notes[(r["speech_id"], r["run"])] = r["note"]
    done = set()
    if Path(a.out).exists():
        done = {(r["speech_id"], r["pair"]) for r in map(json.loads, Path(a.out).read_text(encoding="utf-8").splitlines())}
    with open(a.out, "a", encoding="utf-8") as f:
        for i, sid in enumerate(sorted(ref)):
            for x, y in PAIRS:
                pair = f"{x}-{y}"
                if (sid, pair) in done or (sid, x) not in notes or (sid, y) not in notes:
                    continue
                order = [x, y]
                random.Random(f"{sid}-{pair}").shuffle(order)
                t0 = time.time()
                res = judge(ref[sid], notes[(sid, order[0])], notes[(sid, order[1])], a.model)
                f.write(json.dumps({"speech_id": sid, "pair": pair, "A": order[0], "B": order[1], "model": a.model,
                                    "labels": {order[0]: res["A"], order[1]: res["B"]}, "seconds": round(time.time() - t0, 1)},
                                   ensure_ascii=False) + "\n")
                f.flush()
            if i % 25 == 0:
                print(i, sid, flush=True)

if __name__ == "__main__":
    main()
