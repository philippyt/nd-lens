import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path

HOST = "http://127.0.0.1:11434"
MODEL = "hf.co/NbAiLab/borealis2-26b-a4b-preview-gguf:Q3_K_M"
OPTIONS = {"temperature": 0.0, "seed": 0, "num_ctx": 4096}
BASE = (
    "Du retter feil i automatiske transkripsjoner av spontan, muntlig norsk tale. Transkripsjonen er på bokmål, med små "
    "bokstaver, uten tegnsetting og med tall skrevet med bokstaver. Talerne snakker dialekt, men transkripsjonen skal "
    "være på bokmål. Ikke legg til innhold som ikke er sagt, ikke fjern innhold, ikke omformuler, og ikke gjør teksten "
    "mer formell eller grammatisk korrekt. Behold muntlige trekk som gjentakelser og avbrutte setninger. Hvis du er "
    "usikker, la teksten stå som den er."
)
SYSTEM = {
    "a": BASE + " Rett bare ord som tydelig er feilhørt, altså ord som ikke gir mening i sammenhengen, til det taleren "
         "mest sannsynlig sa. Svar med hele den rettede transkripsjonen.",
    "b": BASE + " Ord som talegjenkjenningen var usikker på står i hakeparentes, slik: [ord]. Steder der et ord kan ha "
         "falt ut er markert med [_]. De fleste markerte ordene er riktige, og de fleste [_] skjuler ingenting. Du kan "
         "bare endre ord i hakeparentes, og bare sette inn ord der det står [_], og bare når det tydelig mangler noe. Alt "
         "annet skal stå uendret. Svar med hele transkripsjonen uten hakeparenteser.",
}

def prompt_hash() -> str:
    return hashlib.sha256(json.dumps({"model": MODEL, "options": OPTIONS, "system": SYSTEM}, sort_keys=True).encode()).hexdigest()[:16]

def chat(system: str, user: str, schema: dict) -> dict:
    body = {"model": MODEL, "stream": False, "think": False, "format": schema, "options": OPTIONS,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    req = urllib.request.Request(f"{HOST}/api/chat", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as f:
        return json.loads(json.load(f)["message"]["content"])

def request(item: dict, variant: str) -> tuple[str, dict]:
    ctx = f"Forrige del av samme opptak (bare som kontekst, skal ikke rettes):\n{item['context']}\n\n" if item["context"] else ""
    schema = {"type": "object", "properties": {"tekst": {"type": "string"}}, "required": ["tekst"]}
    if variant == "a":
        return ctx + f"Transkripsjon som skal rettes:\n{item['text']}", schema
    return ctx + f"Transkripsjon med markerte steder:\n{item['marked']}", schema

def main():
    p = argparse.ArgumentParser()
    p.add_argument("items")
    p.add_argument("out")
    p.add_argument("--segs", help="json list of segment indices to run (pilot)")
    a = p.parse_args()
    print("model", MODEL, "prompt hash", prompt_hash(), flush=True)
    items = [json.loads(l) for l in Path(a.items).read_text(encoding="utf-8").splitlines()]
    if a.segs:
        keep = set(json.loads(Path(a.segs).read_text()))
        items = [x for x in items if x["seg"] in keep]
    done = set()
    if Path(a.out).exists():
        done = {(r["id"], r["variant"]) for r in map(json.loads, Path(a.out).read_text(encoding="utf-8").splitlines())}
    t_start, n = time.time(), 0
    with open(a.out, "a", encoding="utf-8") as f:
        for i, item in enumerate(items):
            for variant in ("a", "b"):
                if (item["id"], variant) in done or (variant == "a" and not item["text"]) or (variant == "b" and not item["slots"]):
                    continue
                user, schema = request(item, variant)
                t0 = time.time()
                try:
                    res, err = chat(SYSTEM[variant], user, schema), None
                except Exception as e:
                    res, err = None, repr(e)
                f.write(json.dumps({"id": item["id"], "variant": variant, "model": MODEL, "prompt_hash": prompt_hash(),
                                    "response": res, "error": err, "seconds": round(time.time() - t0, 1)}, ensure_ascii=False) + "\n")
                f.flush()
                n += 1
            if i % 20 == 0:
                print(f"{i}/{len(items)} items, {n} calls, {time.time() - t_start:.0f} s", flush=True)

if __name__ == "__main__":
    main()
