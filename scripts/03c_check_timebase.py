import numpy as np
import pandas as pd
import soundfile as sf

from dialect_probing.config import load_config
from dialect_probing.nbtale import wav_path

def energy_bounds(audio: np.ndarray, sr: int, win: float = 0.01, rise_db: float = 20.0) -> tuple[float, float]:
    n = int(sr * win)
    frames = audio[: len(audio) // n * n].reshape(-1, n)
    db = 10 * np.log10((frames.astype(np.float64) ** 2).mean(1) + 1e-12)
    loud = np.flatnonzero(db > np.percentile(db, 10) + rise_db)
    return loud[0] * win, (loud[-1] + 1) * win

def main():
    cfg = load_config()
    nb = cfg["data_dir"] / "nbtale"
    utts = pd.read_parquet(nb / "utterances.parquet")
    rows = []
    for uid in utts["utt_id"]:
        info = sf.info(str(wav_path(nb / "raw", uid)))
        rows.append({"utt_id": uid, "sr": info.samplerate, "channels": info.channels, "wav_seconds": info.frames / info.samplerate})
    utts = utts.merge(pd.DataFrame(rows), on="utt_id")
    utts["end_minus_wav"] = utts["ann_end"] - utts["wav_seconds"]
    print("sample rates:", utts["sr"].value_counts().to_dict(), "| channels:", utts["channels"].value_counts().to_dict())
    print("annotation end - wav duration (s):", utts["end_minus_wav"].describe(percentiles=[.01, .5, .99]).round(4).to_dict())
    print("utterances with |diff| > 20 ms:", (utts["end_minus_wav"].abs() > 0.02).sum())

    sample = utts.sample(12, random_state=cfg["nbtale"]["seed"])
    out = []
    for _, r in sample.iterrows():
        audio, sr = sf.read(str(wav_path(nb / "raw", r["utt_id"])))
        on, off = energy_bounds(audio, sr)
        out.append({"utt_id": r["utt_id"].split("/")[-1], "manual_start": r["speech_start"], "energy_onset": on,
                    "manual_end": r["speech_end"], "energy_offset": off})
    df = pd.DataFrame(out)
    df["onset_diff"] = df["energy_onset"] - df["manual_start"]
    df["offset_diff"] = df["energy_offset"] - df["manual_end"]
    print("\nmanual speech start/end vs. energy onset/offset (s):")
    print(df.round(3).to_string(index=False))
    print(f"\nmedian onset diff {df['onset_diff'].median():+.3f} s, median offset diff {df['offset_diff'].median():+.3f} s")

if __name__ == "__main__":
    main()
