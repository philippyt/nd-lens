# H2 plan: dialect-balanced reader training

**Status: paused (2026-10-17).** Not run on this machine. H1 is supported for norsk-asr (see [PREREG.md](PREREG.md)),
so this is the design to pick up later.

Plan only. It runs only if H1 is supported (see [PREREG.md](PREREG.md)). The throughput numbers marked *estimate* are
not yet measured on this machine.

## Dialect labels

- **Source:** the Stortinget Speech Corpus (SSC) v1.0 from Språkbanken (CC0). Each segment has a `speakers` list with `dialect` (east, west, south, mid, north). The labels are inferred from county of birth and represented counties, using Parsons et al.'s mapping.
- **How labels match the training audio:** norsk-asr trained on `NbAiLab/nb_distil_speech_noconcat_stortinget`, whose ids (`stortinget-YYYYMMDD-HHMMSS_<start>_<end>`) match SSC `audio_path` names. So the labels can be joined onto the exact audio and text pipeline norsk-asr used.
- **Download:** `ssc_v1_0.jsonl` is only distributed inside `ssc_v1_0.tar.gz` (66.4 GB), together with the segment audio (MP3). The whole archive is downloaded to `data/ssc/`. Its audio can be used directly, with the distil shards as a cross-check of the text filters.
- **Label quality:** inferred from birth place, 91.5% agreement with NPSC's manual labels. Only single-speaker segments (`num_speakers == 1`) are used.

## Data

Starting pool: SSC single-speaker segments, after norsk-asr's own filters (no digits, `whisper_wer <= 0.15`).

| Area | Share of SSC segments | Rough usable hours after filters (*estimate*) |
|---|---|---|
| east | 34% | ~900 |
| west | 23% | ~600 |
| mid | 10% | ~250 |
| north | 6% | ~160 |
| south | 4% | ~100 |

Two arms of **300 h** each, drawn from the same pool, with each speaker capped at 5 h in both arms:

- **Natural:** areas in their pool proportions.
- **Balanced:** 60 h per area. South is the binding constraint at roughly 100 h and few speakers.

The same NB Tale and NB Samtale speakers never appear in training: they are different corpora. The test sets are
H1's sets plus norsk-asr's own (FLEURS, NPSC test, NST validation, Stortinget validation). Stortinget validation and
NPSC test meetings are already held out of SSC training by SSC's own split.

**Confound to report:** the balanced arm repeats South and North speakers more. A gap reduction could come from
more speakers rather than from dialect per se. The speaker cap and speaker counts per arm are reported.

## Training

- **Trained:** the reader and CTC head, initialized as in norsk-asr.
- **Frozen:** the encoder, with the released L12, and the layer mix at its released values.
- **Settings:** norsk-asr's: lr 3e-4, warmup 500, 600 s batches, dropout 0.1, 5 epochs (about 1,500 audio-hours per run). Two seeds per arm, so 4 runs. Identical steps and compute in both arms.

**SpecAugment:** norsk-asr masks the log-mel *before* the encoder. There are two ways to handle it:

| Option | How | Cost |
|---|---|---|
| **A. Online encoder (recommended on H100)** | Run the frozen encoder every step, keep the original SpecAugment | No cache. More compute per step, recipe unchanged |
| B. Cached mixed features | Cache the mixed encoder output once, apply time and channel masks to the cached features | Changes the augmentation, the same way in both arms |

Option B changes the recipe the same way for both arms, so the arm comparison stays fair. But the absolute WERs would
not be comparable with released norsk-asr.

## Disk (option B)

The mixed output is 1024-d at 25 frames/s, stored as fp16 (it is layer-normed, so fp16 storage is safe; compute stays
bf16/fp32):

- 300 h × 3,600 × 25 × 1,024 × 2 B ≈ **55 GB** per 300 h.
- The two arms overlap; their union is about 450-500 h, so roughly **85-95 GB**.
- 857 GB is free on the WSL disk.
- Option A needs only the compressed audio, about 15 GB for 500 h of MP3.

## Time and cost

*Estimates* from norsk-asr's H100 log (0.83 audio-h/s for the full recipe, including L12 backward) and step 5g on
this machine (about 180× real time for greedy decoding, batch 1):

| | Rented H100 (RunPod, $3.99/h) | RX 9070 XT (local) |
|---|---|---|
| Data prep: shards, join, sampling | ~0.5 h | ~1 h (download over home line) |
| Encoder caching (option B, ~500 h) | ~0.3 h | ~2-3 h |
| One run, 1,500 audio-h | ~20 min (option A, ~1.3 audio-h/s) | ~2-4 h (option B, *to be benchmarked*) |
| 4 runs | ~1.5 h | ~8-16 h |
| Evaluation of 4 runs | ~0.3 h | ~1 h |
| **Total** | **~2.5-3 h, about $10-12** | **~12-20 h, overnight** |

To pin down the local numbers, I'll benchmark a few hundred reader steps on random 1024-d input on the 9070 XT once
the GPU is free.

**Earlier choice (2026-10-17, before pausing): the local RX 9070 XT.** Memory pressure in WSL during the H1
decoding argues for a rented GPU instead.
- The benchmark decides between option A and option B.
- If the online encoder (A) makes one run longer than about 6 h, use cached features (B). Then report that the augmentation moved after the encoder, the same way in both arms.
- Runs go one at a time overnight. Checkpoints are saved every 2,000 steps so a WSL restart doesn't lose a run.
