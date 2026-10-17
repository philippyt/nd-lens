import json
import re
import subprocess
import zipfile
from collections import Counter
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download
from norsk_asr.text import is_clean, normalize

from dialect_probing.config import load_config

N = 6
KENLM = Path.home() / "kenlm_env"
MARK = re.compile(r"%\S+|<[^>]*>")

def nbtale_grams(annotation_zip) -> tuple[set, int]:
    z = zipfile.ZipFile(annotation_zip)
    grams, words = set(), 0
    for part in ("part_1", "part_2", "part_3"):
        block = []
        for line in z.read(f"Annotation/{part}.trans").decode("utf-8", "replace").splitlines() + ["#!TRANSCRIPTION!#"]:
            if line.startswith("#!TRANSCRIPTION!#"):
                w = normalize(" ".join(block)).split()
                words += len(w)
                for n in range(4, N + 1):
                    grams.update(tuple(w[i:i + n]) for i in range(len(w) - n + 1))
                block = []
                continue
            c = line.split("\t")
            if len(c) > 3 and not c[3].startswith("<"):
                block.append(c[3].split(";")[0].replace("_", " "))
    return grams, words

def samtale(root) -> list[str]:
    out = []
    for meta in sorted((root / "hf" / "data").glob("*_metadata.jsonl")):
        for line in meta.read_text().splitlines():
            r = json.loads(line)
            if r["orthography"] == "bm":
                out.append(r["orthographic"])
    return out

def npsc_train() -> list[str]:
    files = [f for f in HfApi().list_repo_files("NbAiLab/NPSC", repo_type="dataset") if re.match(r"data/train/\d+\.json$", f)]
    out = []
    for f in files:
        for line in open(hf_hub_download("NbAiLab/NPSC", f, repo_type="dataset"), encoding="utf-8"):
            r = json.loads(line)
            if r["sentence_language_code"] == "nb-NO":
                out.append(r["sentence_text"])
    return out

def main():
    cfg = load_config()
    lm_dir = cfg["data_dir"] / "lm_conv"
    lm_dir.mkdir(exist_ok=True)
    grams, words = nbtale_grams(cfg["data_dir"] / "nbtale" / "raw" / "Annotation.zip")
    print(f"NB Tale modules 1-3: {words:,} words, {len(grams):,} 4-6-grams to exclude")
    counts, kept = Counter(), []
    for name, sents in (("NB Samtale (Bokmål, all splits)", samtale(cfg["data_dir"] / "nb_samtale")), ("NPSC train (Bokmål, verbatim)", npsc_train())):
        n_in, dropped, bad = len(sents), 0, 0
        for s in sents:
            t = normalize(MARK.sub(" ", s))
            w = t.split()
            if len(w) < 2 or not is_clean(t):
                bad += 1
                continue
            if any(tuple(w[i:i + N]) in grams for i in range(len(w) - N + 1)) or (4 <= len(w) < N and tuple(w) in grams):
                dropped += 1
                continue
            kept.append(t)
            counts.update(w)
        print(f"{name}: {n_in:,} sentences, {bad:,} empty or unclean, {dropped:,} dropped for NB Tale overlap")
    text = lm_dir / "conv.txt"
    text.write_text("\n".join(kept) + "\n")
    (lm_dir / "conv4.unigrams.txt").write_text("\n".join(w for w, _ in counts.most_common()) + "\n")
    print(f"kept {len(kept):,} sentences, {sum(counts.values()):,} words, {len(counts):,} types")
    env = {"LD_LIBRARY_PATH": str(KENLM / "lib"), "PATH": "/usr/bin:/bin"}
    subprocess.run([str(KENLM / "kbin" / "lmplz"), "-o", "4", "--discount_fallback", "-S", "20%", "-T", str(lm_dir),
                    "--text", str(text), "--arpa", str(lm_dir / "conv4.arpa")], check=True, env=env, capture_output=True)
    subprocess.run([str(KENLM / "kbin" / "build_binary"), "-q", "8", "-b", "7", "trie", str(lm_dir / "conv4.arpa"),
                    str(lm_dir / "conv4.bin")], check=True, env=env, capture_output=True)
    (lm_dir / "conv4.arpa").unlink()
    m3 = [w for r in pd.read_parquet(cfg["data_dir"] / "spont" / "segments.parquet").query("set == 'nbtale_m3'")["ref"] for w in r.split()]
    print(f"wrote {lm_dir / 'conv4.bin'}; module 3 reference word tokens covered by its vocabulary: {sum(w in counts for w in m3) / len(m3):.1%}")

if __name__ == "__main__":
    main()
