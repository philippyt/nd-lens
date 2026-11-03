import json
import random
import shutil
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.constants import HF_HUB_CACHE
from norsk_asr.prep import ST_REPO, npsc_days, pack_distil_shard, pack_npsc_day

from dialect_probing.config import load_config

HOURS = 15.0
MAX_WHISPER_WER = 0.15

def drop_download(repo: str):
    """Packed shards hold the PCM; the downloaded archives are not needed again (disk is tight)."""
    shutil.rmtree(Path(HF_HUB_CACHE) / f"datasets--{repo.replace('/', '--')}" / "blobs", ignore_errors=True)
    shutil.rmtree(Path(HF_HUB_CACHE) / f"datasets--{repo.replace('/', '--')}" / "snapshots", ignore_errors=True)

def hours(out: Path) -> float:
    return sum(json.loads(l)["n"] for p in out.glob("*.jsonl") for l in open(p)) / 16000 / 3600

def main():
    cfg = load_config()
    root = cfg["out_dir"] / "style"
    verb, edit = root / "verbatim", root / "edited"
    rng = random.Random(0)

    days = npsc_days("train")
    rng.shuffle(days)
    for d in days:
        if hours(verb) >= HOURS:
            break
        r = pack_npsc_day((d, verb, None))
        drop_download("NbAiLab/NPSC")
        print(json.dumps(r), flush=True)
    target = hours(verb)

    shards = sorted(f for f in HfApi().list_repo_files(ST_REPO, repo_type="dataset") if f.endswith(".jsonl") and "/train" in f)
    rng.shuffle(shards)
    for f in shards:
        if hours(edit) >= target:
            break
        r = pack_distil_shard((ST_REPO, f, edit, MAX_WHISPER_WER, False, None))
        drop_download(ST_REPO)
        print(json.dumps(r), flush=True)
    # trim the last edited shard's index so both arms have the same hours (its PCM file keeps the unused tail)
    last = max(edit.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
    rows = [json.loads(l) for l in open(last)]
    excess = (hours(edit) - target) * 3600 * 16000
    while rows and excess > 0:
        excess -= rows.pop()["n"]
    last.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"verbatim {hours(verb):.2f} h, edited {hours(edit):.2f} h")

if __name__ == "__main__":
    main()
