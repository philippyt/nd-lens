import json
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests
from huggingface_hub import HfApi, hf_hub_url
from norsk_asr.text import has_digits, is_clean, normalize

from dialect_probing.config import load_config

REPO = "NbAiLab/nb_distil_speech_noconcat_stortinget"
MAX_WHISPER_WER = 0.15

def index_shard(name: str) -> list[dict]:
    url = hf_hub_url(REPO, name, repo_type="dataset")
    rows = []
    for _ in range(3):
        try:
            with requests.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    d = json.loads(line)
                    text = d.get("text") or ""
                    wer = d.get("whisper_wer")
                    wer = float(wer) if wer not in (None, "None") else None
                    kept = not has_digits(text) and not (wer is not None and wer > MAX_WHISPER_WER) and is_clean(normalize(text))
                    rows.append({"id": d["id"], "seconds": float(d["audio_duration"]) / 1000, "text": text,
                                 "whisper_wer": wer, "kept": kept, "shard": name})
            return rows
        except Exception:
            rows = []
    raise RuntimeError(name)

def main():
    cfg = load_config()
    out = cfg["data_dir"] / "train_index"
    out.mkdir(parents=True, exist_ok=True)
    shards = sorted(f for f in HfApi().list_repo_files(REPO, repo_type="dataset") if f.startswith("data/train-"))
    done = {p.stem for p in out.glob("*.parquet")}
    todo = [s for s in shards if s.split("/")[-1].split(".")[0] not in done]
    print(f"{len(shards)} shards, {len(todo)} to index", flush=True)
    with ThreadPoolExecutor(8) as ex:
        for i, (name, rows) in enumerate(zip(todo, ex.map(index_shard, todo))):
            pd.DataFrame(rows).to_parquet(out / f"{name.split('/')[-1].split('.')[0]}.parquet")
            if i % 50 == 0:
                print(i, name, flush=True)
    df = pd.concat([pd.read_parquet(p) for p in sorted(out.glob("*.parquet"))])
    df["date"] = df["id"].str.extract(r"stortinget-(\d{8})")[0]
    k = df[df["kept"]]
    print(f"segments {len(df)}, kept {len(k)} ({k['seconds'].sum() / 3600:.0f} h of {df['seconds'].sum() / 3600:.0f} h), "
          f"dates {df['date'].nunique()} ({df['date'].min()}-{df['date'].max()})")

if __name__ == "__main__":
    main()
