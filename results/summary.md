# Step 5 findings

Speakers in groups 1-12 are split in half within each group and gender. The selection half (130
speakers) picks a pooling lag per layer; every number below is from the other half (120 speakers in
groups 1-12, 26,272 word tokens, 448 word types read in at least 3 groups). Same-speaker and
same-sentence pairs are excluded. Main windows are the manual word boundaries shifted by the
selected lag; unshifted manual and shifted CTC windows are robustness checks.

## Lag per layer

![](lag_per_layer.png)

The lag is chosen by a word identity probe (118 words, 2 speaker folds), not by any dialect contrast.
It grows with depth: 0 s up to L2, 0.12 s at L5-L7, 0.24 s at L11-L12. Without the shift, word
identity accuracy at L12 falls from 0.95 to 0.68, so layer-matched windows matter.

## Convergence

![](similarity.png) ![](convergence_controls.png)

Same-word similarity across dialect groups stays close to similarity within a group at every layer
(across / within ≥ 0.92). The dialect gap is not significant up to L3, grows through the middle
layers, peaks at L8 (ratio 0.926, gap 0.039, speaker-permutation p ≤ 0.01) and shrinks toward the
top (L12 0.984, still p ≤ 0.01). Upper layers partly converge, but they don't fully converge.
The same shape holds with NTNU recordings only, on set A sentences only (identical text for all
speakers), and with unshifted or CTC windows.

| Layer | Within | Across | Ratio | Dialect probe |
|---|---|---|---|---|
| L0 | 0.225 | 0.225 | 0.999 | 0.073 |
| L4 | 0.412 | 0.394 | 0.957 | 0.116 |
| L8 | 0.523 | 0.485 | 0.926 | 0.219 |
| L12 | 0.619 | 0.610 | 0.984 | 0.200 |
| L12 orig | 0.638 | 0.627 | 0.983 | 0.201 |

## Dialect probe

![](dialect_probe.png)

A 12-way linear probe on single word tokens (5 speaker folds) is at or below the majority baseline
(0.088) up to L2. It rises to 0.22 at L7-L9 and drops slightly to 0.20 at L12. It is
not explained by recording place: with NTNU speakers only it reaches 0.216.

## Strata

![](convergence_strata.png)

- **Frequency:** no consistent effect.
- **Length:** words of 7+ letters diverge more at every layer.
- **Words with r but no r+consonant cluster:** these show the largest dialect gap at all layers, including L0 (ratio 0.86-0.94). This fits the tapped vs uvular r split.
- **Retroflex contexts:** close to no gap early, then a clear dip at L5-L8 (0.90).
- **Pronoun and negation markers:** only 12 types; they behave like controls.
- **Infinitive class:** it holds words like *dette*, *bare*, *neste* that are rarely used as infinitives, so it is not a clean test of -a/-e endings.

## Error rate and distance from East

![](wer_distance.png) ![](wer_distance_scatter.png)

Distance from East is 1 minus the mean same-word similarity to groups g11 and g12. Across the 12 groups it tracks
greedy Bokmål WER from L2 upward, strongest at L8 (Spearman 0.94, permutation p < 0.001). Leaving
out g11 and g12 themselves, which are near by construction, it still holds (n = 10, ρ = 0.95 at L8).
g08 (Rogaland and Hordaland) is recorded entirely at NTNU and has both the largest distance and the
highest WER, so recording place does not drive this.

## Original vs fine-tuned layer 12

Fine-tuning only L12 left it nearly unchanged on every measure:

| Measure | Fine-tuned | Original |
|---|---|---|
| Selected lag | 0.24 s | 0.24 s |
| Word probe | 0.949 | 0.949 |
| Across / within ratio | 0.984 | 0.983 |
| Dialect probe | 0.200 | 0.201 |
| WER correlation (ρ) | 0.83 | 0.84 |

The partial convergence at the top is already in Google's encoder. Fine-tuning did not add it.

## Caveats

- NB Tale modules 1-2 are read speech. Module 3 (spontaneous speech) is a planned follow-up.
- WER and CER are greedy without a language model, and only Bokmål prompts are used. Nynorsk prompts have about 50% WER for every group, because the model only writes Bokmål.
- 12 groups is a small n for the correlation. Groups g01-g04, g07, g09 and g10 include speakers recorded at UiT or UiA.
- With 100 permutations, p cannot go below 0.01.
