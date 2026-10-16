# Preregistration: dialect gap on spontaneous speech

Written before any spontaneous-speech audio is decoded. Changes after the commit that adds this file are listed under
[Deviations](#deviations), with date and reason.

## Questions

**H1.** On spontaneous speech, do Norwegian ASR models make clearly more errors for western and southwestern speakers
than for eastern ones?

**H2.** If so, does training the norsk-asr reader on a dialect-balanced mix shrink that gap without meaningfully raising
overall WER?

H2 is run only if H1 finds a meaningful gap for norsk-asr greedy on the primary test set (defined below), since H2
retrains that model.

## Context from read speech

Steps 1-5 used NB Tale modules 1-2 (read prompts) and norsk-asr greedy.

**The region-level gap:**
- West (g06-g09) vs East (g11-g12) Bokmål WER: 24.1% vs 20.7%. The gap is +3.4 points (+16% relative), with a speaker bootstrap 95% CI of [+1.3, +5.4].
- For substitutions only, the gap is +2.4 points (+15% relative).

**Inside the encoder:**
- Word representations converge across dialects only partly, but cross-dialect similarity stays within 8% of within-dialect similarity.
- Token-level errors follow general atypicality, not distance from Eastern pronunciations.

Read speech constrains the words and largely the pronunciation. So we expect a gap of similar size on spontaneous
speech, perhaps larger from dialect vocabulary and morphology. We do not expect a dramatic one.

## Test sets

Only native speakers. Nothing from these sets is used for training, tuning or LM building.

1. **NB Tale module 3 (primary).**
   - Content: free speech, about 2 minutes per speaker; 365 recordings, 12.7 h in total.
   - Download: `sennheiser_3.tar.gz`, 3.3 GB, Språkbanken. Annotations are in `Annotation.zip` (`part_3.xml`).
   - Speakers: the native speakers of modules 1-2 (groups g01-g12 and g24), with the same speaker metadata.
   - References: the transcriptions are standardized Bokmål. Dialect forms are tagged `slang` with a normalized form (*kjem* → *kommer*), and the normalized form is used.
   - Segmentation: recordings are cut at the annotated sentence boundaries into segments of at most 30 s. Adjacent sentences are merged while the total stays ≤ 30 s. Single sentences over 30 s are dropped and counted.
   - Dropped segments: any containing `<incomprehensible>` or `<anonymized>`. Fillers, breaths and other non-word events are not part of the reference.
2. **NB Samtale (secondary).**
   - Content: conversational speech from podcasts and National Library live events. 69 speakers, 24 h in total.
   - Download: Hugging Face `Sprakbanken/nb_samtale`, about 2.4 GB in total, of which about 1.8 GB is Bokmål.
   - Segments: all splits are used. Only segments with Bokmål orthography are kept, using the `orthographic` transcription (standardized word forms).
   - Dialect field: e, n, sw, t, w. Eastern speakers make up about half of the speech time.

### Dialect areas

| Area | NB Tale module 3 | NB Samtale |
|---|---|---|
| **West (incl. southwest)** | g06, g07, g08, g09 | w, sw |
| **East** | g11, g12 | e |
| North | g01, g02, g03 | n |
| Central | g04, g05 | t |
| South | g10 | - |

The primary contrast is West vs East. All areas are reported. g24 (mixed origin) counts toward overall WER only.

## Models

| Model | Decoding | Training data (as documented) |
|---|---|---|
| norsk-asr `full_top1` | greedy (**primary**) | Stortinget, NST, NPSC (1,680 h); encoder pretrained by Google, data unknown |
| norsk-asr `full_top1` | beam + 4-gram KenLM | as above; LM text from NCC |
| NbAiLab/nb-whisper-large | greedy (beam 1), `language=no`, `task=transcribe` | ncc_speech (NRK subtitles, Stortinget, audiobooks), NST, NPSC; ~66k h |
| NbAiLab/nb-whisper-small | as above | as above |
| NbAiLab/nb-wav2vec2-300m-bokmaal | greedy CTC, no LM | fine-tuned on NPSC Bokmål; base KBLab/wav2vec2-large-voxrex (Swedish) |

**Contamination:**
- **NB Tale** is not in any documented training set.
- **NB Samtale** is not listed for any model. It came out in 2023, after NCC was built.
- **ncc_speech** is gated and its full source list is not public, so National Library live events can't be ruled out for NB-Whisper.
- **Checks:**
  - 6+ word reference sentences found verbatim in any training or LM text we can access.
  - The rate of exact-match segments per model and set, compared between NB Tale module 3 and NB Samtale.
  - A set with an outlying exact-match rate for one model is flagged, and that model's result on it is reported as possibly contaminated.

## Metrics

- **Normalization:** hypotheses and references go through `norsk_asr.text.normalize`. It lowercases, strips punctuation and spells out numbers. Hyphens split words.
- **WER and CER per area:** errors pooled over the area's speakers, divided by its reference words (or characters).
- **Misheard words:** substitutions whose hypothesis word is not a form of the same lemma as the reference word, as a share of reference words. Lemmas come from the Norsk ordbank Bokmål full-form list. A substitution between forms of the same lemma (*jobba* / *jobbet*) does not count. Deletions and insertions are left out, since they mostly reflect transcription style (fillers, clean-up).
- **Uncertainty:** speakers are resampled with replacement within each area, 10,000 times, seed 0. Percentile 95% CIs are given for each area's rates and for the West-East gap (absolute and relative).

## What counts as a meaningful effect

**H1 (per model and test set):** a meaningful gap needs all three of:
1. West WER is at least **20% higher** (relative) than East WER;
2. the 95% CI of the absolute WER gap excludes 0;
3. the 95% CI of the misheard-word gap excludes 0, in the same direction.

For reference, read speech gave +16% relative, just below threshold (1).

H1 is **supported** if the primary model (norsk-asr greedy) on the primary set (NB Tale module 3) meets all three, and
NB Samtale shows a gap in the same direction for the same model. NB Samtale has few speakers per area, so its CI need
not exclude 0. Results for the other models are descriptive.

**H2:** dialect-balanced training counts as meaningful only if both hold:
- it cuts the West-East WER gap by at least **30% relative** compared with natural-mix training (paired speaker bootstrap; the CI of the reduction excludes 0);
- overall WER rises by no more than **3% relative**, both on the pooled spontaneous test sets and averaged over norsk-asr's own test sets (FLEURS, NPSC, NST, Stortinget validation).

## H2 design

Run only if H1 is supported. It is planned in detail in step 9.

- **What is retrained:** only the reader. The encoder (with the released L12) and the layer mix are frozen at released values. The reader and CTC head start from the same initialization.
- **Data:** about a 300 h Stortinget subset, using the speaker dialect metadata of Parsons et al. (2025), inferred from municipality of birth.
- **Arms:** natural dialect mix vs balanced across the five areas. Same hours, steps, hyperparameters and seeds.
- **Seeds:** two per arm.
- **Evaluation:** greedy, as in H1.

## Deviations

None yet.
