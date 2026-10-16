import time

import jiwer
import numpy as np
import pandas as pd
import torch
from norsk_asr.evaluate import greedy
from norsk_asr.text import has_digits
from norsk_asr.train import load_checkpoint
from norsk_asr.transcribe import load_audio

from dialect_probing.align import FRAME_SECONDS, ctc_viterbi, log_probs, word_spans
from dialect_probing.config import load_config
from dialect_probing.nbtale import wav_path

def pick_speakers(spk: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    spk = spk[~spk["group"].isin(cfg["analysis"]["exclude_groups"])]
    n = cfg["nbtale"]["speakers_per_group"]
    return spk.groupby("group", group_keys=False).apply(lambda g: g.sample(min(n, len(g)), random_state=cfg["nbtale"]["seed"]))

def main():
    cfg = load_config()
    nb = cfg["data_dir"] / "nbtale"
    out = cfg["out_dir"] / "align"
    out.mkdir(parents=True, exist_ok=True)
    spk = pick_speakers(pd.read_parquet(nb / "speakers.parquet"), cfg)
    utts = pd.read_parquet(nb / "utterances.parquet")
    utts = utts[utts["speaker"].isin(spk["speaker"])]
    digits = utts["text"].map(has_digits)
    print(f"{len(spk)} speakers, {len(utts)} utterances, {digits.sum()} skipped for digits")
    utts = utts[~digits]
    words = pd.read_parquet(nb / "ref_words.parquet")
    words = words[words["utt_id"].isin(utts["utt_id"])].sort_values(["utt_id", "word_idx"])

    device = torch.device("cuda")
    model, units = load_checkpoint(cfg["checkpoint"], device)
    word_rows, utt_rows = [], []
    t0, audio_s = time.time(), 0.0
    for uid, w in words.groupby("utt_id", sort=False):
        audio = load_audio(str(wav_path(nb / "raw", uid)))
        audio_s += len(audio) / 16000
        lp = log_probs(model, audio, device)
        hyp = units.decode(greedy(torch.from_numpy(lp)[None], torch.tensor([len(lp)]))[0])
        ids = [units.encode(x) for x in w["word"]]
        token_word = np.concatenate([[i] * len(x) for i, x in enumerate(ids)]).astype(int)
        res = ctc_viterbi(lp, [t for x in ids for t in x])
        ref = " ".join(w["word"])
        utt_rows.append({"utt_id": uid, "ref": ref, "hyp": hyp, "frames": len(lp), "aligned": res is not None,
                         "align_logprob_per_frame": res[1] / len(lp) if res else np.nan})
        if res is None:
            continue
        for (_, row), span in zip(w.iterrows(), word_spans(res[0], token_word, len(w))):
            word_rows.append({**row.to_dict(), "ctc_start_frame": span[0], "ctc_end_frame": span[1],
                              "ctc_start": span[0] * FRAME_SECONDS, "ctc_end": span[1] * FRAME_SECONDS})
    took = time.time() - t0

    uw = pd.DataFrame(utt_rows)
    ww = pd.DataFrame(word_rows)
    uw.to_parquet(out / "utterances.parquet")
    ww.to_parquet(out / "words.parquet")
    print(f"{audio_s / 3600:.2f} h audio in {took:.0f} s ({audio_s / took:.0f}x real time); "
          f"aligned {uw['aligned'].sum()}/{len(uw)} utterances, {len(ww)} words")

    ww["start_diff"] = ww["ctc_start"] - ww["ref_start"]
    ww["end_diff"] = ww["ctc_end"] - ww["ref_end"]
    inter = (np.minimum(ww["ctc_end"], ww["ref_end"]) - np.maximum(ww["ctc_start"], ww["ref_start"])).clip(lower=0)
    ww["ref_covered"] = inter / (ww["ref_end"] - ww["ref_start"])
    ww["ctc_in_ref"] = inter / (ww["ctc_end"] - ww["ctc_start"])
    ww["ctc_frames"] = ww["ctc_end_frame"] - ww["ctc_start_frame"]
    ww["ref_frames"] = (ww["ref_end"] - ww["ref_start"]) / FRAME_SECONDS
    q = lambda s: f"{s.median():+.3f} [{s.quantile(.25):+.3f}, {s.quantile(.75):+.3f}]"
    print("\nCTC vs manual word boundaries (seconds, median [IQR]):")
    print("  start diff", q(ww["start_diff"]))
    print("  end diff  ", q(ww["end_diff"]))
    print(f"  ref span covered by CTC span: median {ww['ref_covered'].median():.2f}, mean {ww['ref_covered'].mean():.2f}")
    print(f"  CTC span inside ref span:     median {ww['ctc_in_ref'].median():.2f}, mean {ww['ctc_in_ref'].mean():.2f}")
    print(f"  frames per word: CTC median {ww['ctc_frames'].median():.0f}, manual median {ww['ref_frames'].median():.1f}")
    print(f"  words with < {cfg['alignment']['min_word_frames']} CTC frames: {(ww['ctc_frames'] < cfg['alignment']['min_word_frames']).mean():.1%}")

    per_group = ww.groupby("group").agg(words=("word", "size"), start_diff=("start_diff", "median"),
                                        abs_start_diff=("start_diff", lambda s: s.abs().median()),
                                        ref_covered=("ref_covered", "mean"), ctc_frames=("ctc_frames", "median"))
    uw = uw.merge(utts[["utt_id", "group"]], on="utt_id")
    per_group["wer"] = uw.groupby("group").apply(lambda g: jiwer.wer(list(g["ref"]), list(g["hyp"])))
    per_group["cer"] = uw.groupby("group").apply(lambda g: jiwer.cer(list(g["ref"]), list(g["hyp"])))
    print("\nper dialect group:")
    print(per_group.round(3).to_string())
    print(f"\noverall greedy WER {jiwer.wer(list(uw['ref']), list(uw['hyp'])):.3f}, CER {jiwer.cer(list(uw['ref']), list(uw['hyp'])):.3f}")

if __name__ == "__main__":
    main()
