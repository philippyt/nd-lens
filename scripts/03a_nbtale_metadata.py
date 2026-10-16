import pandas as pd
from norsk_asr.text import normalize

from dialect_probing.config import load_config
from dialect_probing.nbtale import read_annotations, read_speakers

def main():
    cfg = load_config()
    out = cfg["data_dir"] / "nbtale"
    raw = out / "raw"
    spk = read_speakers(raw)
    utts, words = read_annotations(raw)

    if cfg["nbtale"]["native_only"]:
        spk = spk[spk["native"]]
    utts = utts[utts["speaker"].isin(spk["speaker"])].merge(spk[["speaker", "region", "macro_region"]], on="speaker")
    words = words.merge(utts[["utt_id", "speaker", "group", "macro_region", "sentence_set"]], on="utt_id")
    words["word"] = words["word_orig"].map(normalize)
    words = words[words["word"].str.len() > 0]

    spk.to_parquet(out / "speakers.parquet")
    utts.to_parquet(out / "utterances.parquet")
    words.to_parquet(out / "ref_words.parquet")

    print(f"native speakers: {len(spk)}, utterances: {len(utts)}, word tokens: {len(words)}")
    print(f"reference audio (to annotation end): {utts['ann_end'].sum() / 3600:.2f} h\n")
    print("speakers per dialect group:")
    print(spk.groupby(["group", "region", "macro_region"]).agg(speakers=("speaker", "size"),
          female=("gender", lambda g: (g == "F").sum()), place=("recording_place", lambda p: ",".join(sorted(set(p)))))
          .to_string())

    dial = words[~words["group"].isin(cfg["analysis"]["exclude_groups"])]
    per_word = dial.groupby("word").agg(
        tokens=("speaker", "size"), speakers=("speaker", "nunique"), groups=("group", "nunique"),
        macro=("macro_region", "nunique"), in_set_a=("sentence_set", lambda s: (s == "a").any()))
    per_word = per_word.sort_values(["groups", "speakers"], ascending=False)
    per_word.to_csv(out / "shared_words.csv")

    print("\nword types by number of dialect groups (g01-g12) they occur in:")
    for k in (12, 9, 6, 3):
        sel = per_word[per_word["groups"] >= k]
        print(f"  >= {k:2d} groups: {len(sel):5d} types, {sel['tokens'].sum():6d} tokens")
    sel = per_word[per_word["groups"] == 12]
    print(f"\nall 12 groups, >= 5 speakers in every group: "
          f"{(dial[dial['word'].isin(sel.index)].groupby(['word', 'group'])['speaker'].nunique().unstack().min(axis=1) >= 5).sum()} types")
    print("\ntop 40 words by groups, then speakers:")
    print(per_word.head(40).to_string())
    print("\nset A sentences (read by every speaker):")
    for t in utts[utts["sentence_set"] == "a"].drop_duplicates("sentence_id").sort_values("sentence_id")["text"]:
        print("  ", t)

if __name__ == "__main__":
    main()
