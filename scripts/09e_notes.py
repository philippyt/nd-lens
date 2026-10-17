import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path

HOST = "http://127.0.0.1:11434"
MODEL = "gemma4:12b"
OPTIONS = {"temperature": 0.6, "top_p": 0.9, "num_ctx": 8192}
RUNS = {"ref1": ("reference", 1), "ref2": ("reference", 2), "asr": ("asr", 3)}
FIELDS = {
    "problem": "Problemet eller saken taleren tar opp",
    "fakta": "Fakta og tall taleren viser til, med tall nøyaktig slik de er sagt",
    "standpunkt": "Talerens standpunkt: hva taleren er for eller mot",
    "tiltak": "Tiltak taleren foreslår, krever eller vil gjennomføre fremover (ikke ting som allerede er gjort). Spørsmål er ikke tiltak",
}
SYSTEM = (
    "Du lager korte, strukturerte notater fra muntlige innlegg i Stortinget. Teksten er en automatisk eller manuell "
    "transkripsjon uten tegnsetting. Bruk bare det som faktisk står i teksten. Ikke legg til kunnskap, ikke tolk "
    "utover det som er sagt, og ikke vurder politikken. Behold nektelser og tall nøyaktig. Skriv på bokmål. "
    "Hvis noe ikke er nevnt, skriv \"Ikke nevnt\" (eller en tom liste)."
)
SCHEMA = {
    "type": "object",
    "properties": {
        "problem": {"type": "string"},
        "fakta": {"type": "array", "items": {"type": "string"}},
        "standpunkt": {"type": "string"},
        "tiltak": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["problem", "fakta", "standpunkt", "tiltak"],
}

def prompt_hash() -> str:
    blob = json.dumps({"model": MODEL, "options": OPTIONS, "fields": FIELDS, "system": SYSTEM, "schema": SCHEMA},
                      ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]

def chat(text: str, model: str, seed: int) -> dict:
    prompt = "Felter:\n" + "\n".join(f"- {k}: {v}" for k, v in FIELDS.items()) + f"\n\nTranskripsjon:\n{text}"
    body = {"model": model, "stream": False, "think": False, "format": SCHEMA, "options": {**OPTIONS, "seed": seed},
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]}
    req = urllib.request.Request(f"{HOST}/api/chat", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as f:
        return json.loads(json.load(f)["message"]["content"])

def main():
    p = argparse.ArgumentParser()
    p.add_argument("inp")
    p.add_argument("out")
    p.add_argument("--model", default=MODEL)
    p.add_argument("--limit", type=int)
    a = p.parse_args()
    print("prompt hash", prompt_hash(), flush=True)
    rows = [json.loads(l) for l in Path(a.inp).read_text(encoding="utf-8").splitlines()][: a.limit]
    done = set()
    if Path(a.out).exists():
        done = {(r["speech_id"], r["run"]) for r in map(json.loads, Path(a.out).read_text(encoding="utf-8").splitlines())}
    with open(a.out, "a", encoding="utf-8") as f:
        for i, r in enumerate(rows):
            for run, (source, seed) in RUNS.items():
                if (r["speech_id"], run) in done:
                    continue
                t0 = time.time()
                note = chat(r[source], a.model, seed)
                f.write(json.dumps({"speech_id": r["speech_id"], "run": run, "source": source, "seed": seed, "model": a.model,
                                    "prompt_hash": prompt_hash(), "note": note, "seconds": round(time.time() - t0, 1)},
                                   ensure_ascii=False) + "\n")
                f.flush()
            if i % 25 == 0:
                print(i, r["speech_id"], flush=True)

if __name__ == "__main__":
    main()
