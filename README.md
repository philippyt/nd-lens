# norsk-dialect-probing

Does the audio encoder of a Norwegian speech recognizer map different dialect pronunciations of the
same word to similar internal representations (convergence), or do representations stay clustered
by dialect?

The model is [norsk-asr](https://github.com/ahdrage/norsk-asr) ([weights](https://huggingface.co/ahdrage/norsk-asr)):
the 12-layer audio encoder of [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) with
only the top layer fine-tuned, a learned softmax mix over its 13 hidden states, and a 6-layer
conformer reader trained with CTC over 83 characters. Layers 1-11 are Google's original weights, so
we can look at an encoder that never saw Norwegian ASR training (layers 1-11, and layer 12 as
released by Google) next to the one layer that did.

The data is [NB Tale](https://hdl.handle.net/21.11146/31) (Språkbanken, CC0): read speech from
12 dialect regions, 20 native speakers each, plus 20 native speakers of mixed origin. Every speaker
reads 3 shared calibration sentences (set A), 12 sentences shared with two other speakers (set B)
and 5 unique sentences (set C).

## Related work

Parsons et al. (2023), "A character-based analysis of impacts of dialects on end-to-end Norwegian
ASR" (NoDaLiDa), analyse the *output* errors of Norwegian ASR on NB Tale by dialect region. Here we
look *inside* the model: where in the encoder dialect information lives, and whether it fades.

## Setup

```bash
git clone https://github.com/ahdrage/norsk-asr.git ../norsk-asr
uv sync
uv run python scripts/fetch_checkpoint.py
```

norsk-asr is imported from the clone, never copied. The EmbeddingGemma 2 encoder (1.5 GB) is
downloaded from the Hub on first use. Paths and subset size are in [`config.yaml`](config.yaml).

## Steps

| Script | What |
|---|---|
| `scripts/00_check_env.py` | GPU visible to torch; fp32 / bf16 matmul sanity |
| `scripts/01_layer_weights.py` | learned layer-mix weights from the checkpoint |
| `scripts/03a_nbtale_metadata.py` | speaker, utterance and word tables; speakers per group; words shared across groups; written standard per sentence |
| `scripts/03b_ctc_align.py` | greedy transcripts and CTC forced alignment with the model's own head; compared with the manual word boundaries |
| `scripts/03c_check_timebase.py` | manual timings vs. audio: sample rates, durations, speech onset |
| `scripts/04_extract.py` | word-pooled encoder activations for all 13 hidden states and the original layer 12 |
| `scripts/04b_lag_sweep.py` | same-word similarity as the pooling window is shifted later |
| `scripts/05a_select_lag.py` | split speakers in half; pick each layer's lag on one half with a word identity probe |
| `scripts/05b_convergence.py` | within vs across dialect similarity of the same word, permutation test, controls and strata |
| `scripts/05c_dialect_probe.py` | 12-way dialect group probe on word tokens |
| `scripts/05d_wer_distance.py` | error rates per group; correlation with representational distance from East |
| `scripts/05f_word_errors.py` | per-token distance from the Eastern and overall word centroids vs recognition errors; layer-mix weights vs dialect signal |
| `scripts/05g_mix_ablation.py` | greedy WER per region with layer-mix weights reduced |
| `scripts/05e_plots.py` | figures in `results/` (run last) |

Findings from step 5 are in [`results/summary.md`](results/summary.md).

Scripts that read downloaded data are run with `python -I`.

## Data

NB Tale is downloaded from Språkbanken (`https://www.nb.no/sbfil/talegjenkjenning/lingit/`):
`Annotation.zip`, `Documentation.zip` and the headset recordings `sennheiser_1.tar.gz` (module 1)
and `sennheiser_2.tar.gz` (module 2), unpacked under `data/nbtale/raw`. Only native speakers are
used: groups 1-12 (one dialect region each) and group 24 (mixed regions, left out of dialect
comparisons). Utterances whose prompt contains digits are skipped, as in norsk-asr's training.

Each sentence is tagged Bokmål, Nynorsk or unclear by counting words that exist in only one of the
two Norsk ordbank full-form lists (`https://www.nb.no/sbfil/leksikalske_databaser/ordbank/`,
`20220201_norsk_ordbank_nob_2005.tar.gz` and `20220201_norsk_ordbank_nno_2012.tar.gz`, unpacked
under `data/ordbank`). norsk-asr writes Bokmål only, so error rates are reported per written standard.

## Word windows

The encoder runs at 25 frames/s; frame *j* is centred at 0.04·*j* s. Two kinds of word windows are
used:

- **manual**: NB Tale's hand-segmented word boundaries (frames whose centre falls inside the word);
- **CTC**: the model's own forced alignment, shifted back by the median lag against the manual
  boundaries (4 frames).

Raw CTC spans start about 0.22 s after the word, and deeper layers carry a word later than shallow
ones (`04b_lag_sweep.py`), so the manual window is also shifted later in 0.04 s steps.

## Outputs

`outputs/activations/`:

| File | Shape | What |
|---|---|---|
| `words.parquet` | one row per word | word, speaker, group, region, sentence set, written standard, manual and CTC times and frame windows; row *i* matches index *i* of the arrays |
| `manual.npy` | words × 14 × 1024, float32 | mean over the manual window of each hidden state |
| `ctc_shifted.npy` | words × 14 × 1024, float32 | the same over the shifted CTC window |
| `lag_sweep.npy` | words × 14 × 11 × 1024, float16 | per-frame layer-normed states, averaged over the manual window shifted later by 0-10 frames |
| `ctc_shifted_ln.npy` | words × 14 × 1024, float16 | per-frame layer-normed states over the shifted CTC window |
| `meta.json` | | layer names and versions, frame length, CTC lag, sweep layers and shifts |

The 14 layers are the 13 hidden states of the released model (`L0` is the input after subsampling,
`L1`-`L12` the conformer layers; version `finetuned`) and Google's original layer 12 (`L12_orig`,
version `original`). Layers 0-11 are identical in both models.

`outputs/align/` holds the greedy transcripts and per-word alignments. `outputs/step5/` holds the
speaker split, selected lags and the tables behind the figures.
