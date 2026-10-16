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
| `scripts/03a_nbtale_metadata.py` | speaker, utterance and word tables; speakers per group; words shared across groups |

Scripts that read downloaded data are run with `python -I`.
