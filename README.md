# VLM Probing

Controlled stress sweeps (density, occlusion) on vision-language models, paired with layer-wise probing of internal representations. Act 1 is a mild first pass toward extreme cases; genuinely extreme cases (counterfactually edited illusions) are the planned next step.

**Status:** Act 1 (counting / occlusion) is complete and preliminary. It largely replicates existing findings, so I treat it as a calibration step and the reason for moving to Act 2.

## Motivation

Standard benchmarks give a single accuracy number. A more informative approach is to construct extreme, tightly controlled visual cases (one variable changes at a time) and observe *how* models fail. This repo asks a mechanistic version of that question:

> When a VLM gives a wrong answer on an extreme case, is the correct information absent from its internal representations, or present but not reflected in the output?

## Research question

When a VLM fails on a controlled but hard visual case, which of these is the cause?

1. **Encoding failure:** the relevant information never appears in the model's internal representations.
2. **Output-stage misalignment:** the information is present internally but is not reflected in the final answer.
3. **Prior/template override:** the answer is driven by a learned layout or language prior more than by the specific image.

### Act 1: counting and occlusion (done, preliminary)

- **Hypothesis:** if failures are output-stage, a linear probe should recover the true count from hidden states even on items where the model's spoken answer is wrong.
- **Would count against it:** probe error at or near the shuffled-label control on the failure cases.
- **Outcome:** probes beat the shuffled control at every layer in both models, so encoding failure is not the main story here. The models differ in the shape of the curve (see Results).

### Act 2: visual illusions with counterfactual edits (planned)

- **Setup:** classic illusions (e.g. Ebbinghaus, Muller-Lyer, checkerboard brightness) at graded edit strengths, from the original illusion to a fully edited image where the illusion no longer holds, plus matched no-inducer controls.
- **Question:** when a model keeps giving the illusory answer after an edit that should change it, which of the three failure types above is it?
- **Predictions:**
  - Encoding failure: probes for the post-edit ground truth stay near chance at all layers.
  - Output-stage misalignment: probes rise above chance in mid/late layers while the spoken answer stays wrong, and patching activations from a matched control run should flip the answer.
  - Prior/template override: behavior is insensitive to edit strength across models and illusion types, with a weaker or inconsistent probe signal.
- **Would count against the output-stage account:** patching at the layer flagged by probing does not change the answer more often than patching a random layer.

### What this does not claim

Probing shows information is decodable, not that the model uses it. Causal claims need intervention experiments, which are planned for Act 2 and not done.

## Setup

- **Models:** Qwen2-VL-7B-Instruct and LLaVA-1.6-7B (Mistral), both 4-bit quantized, run on a Colab T4.
- **Stimuli (41 synthetic images, 512x512, PIL-generated, `data/synthetic/`):**
  - *Density sweep:* N = 2-16 circles, two random layouts per N (30 images).
  - *Size-varied control:* N in {3, 6, 9, 12, 15} with circle radius randomized independently of N, to test for a total-pixel-area shortcut (5 images).
  - *Occlusion sweep:* 8 circles as 4 pairs; pair spacing from fully separated to 75% overlap (6 images).
- **Prompt:** "How many objects are in the image?"
- **Probing:** mean-pooled hidden state per layer -> Ridge regression predicting the true count; leave-one-out CV; compared against a shuffled-label control. Metric: mean absolute error (MAE), lower = more decodable.

## Results

### Behavior

| | Qwen2-VL-7B | LLaVA-1.6-7B |
|---|---|---|
| Low density (N <= 7) | Correct | Correct up to N = 4, then frequent undercounts |
| High density (N >= 8) | Small +/-1-2 errors, reproducible across both layouts | Large jumps to round grid counts (e.g. 9, 12, 16), sometimes with a grid description that contradicts its own number |
| Size-varied control | Correct at N = 3, 6, 9, 12; off by one at 15 | Errors at N = 3, 6, 15 |
| Occlusion | Correct when fully separated (8); answers 6 with partial separation; answers 4 once objects touch and stays at 4 for 0-75% overlap | Answers 4 in every occlusion condition, including fully separated |

### Probing (see `results/`)

- Both models: probe MAE is far below the shuffled-label control at every layer (Qwen 0.24-0.37 vs. control 4.2-4.8; LLaVA 0.31-1.32 vs. control 3.5-4.3).
- **Qwen:** count is decodable from the earliest layers through the last.
- **LLaVA:** decodability is worse early, degrades through the middle layers (worst around layer 12), and only becomes strong in the last few layers.
- In the initial 19-image run, Qwen's probe predicted a count near the true value (about 8) on the occlusion images even though the model's spoken answer was "four" (per-item output in the notebook).

### Interpretation

The count information is present internally for both models, so these errors are not simply a failure to see the objects. The models differ in how they fail: Qwen's errors look like imprecise estimation plus a proximity-based merging of touching objects; LLaVA's look more like defaulting to familiar layout counts. This is consistent with the "the count is there but misaligned" pattern reported in recent work (see `literature_review.md`), so this repo does not claim novelty for the counting result.

### Related Work

- **The Count Is There, but Misaligned: Understanding and Correcting Counting Failures in VLMs** (arXiv:2607.09544, 2026). Layer-wise probing across several VLMs and counting datasets; reports that the correct count is often linearly decodable from internal activations even when the verbalized answer is wrong, and validates causally with activation steering.
- **Counting Circuits: Mechanistic Interpretability of Visual Reasoning in Large Vision-Language Models** (arXiv:2603.18523, 2026). Identifies attention-head roles in counting using activation patching and reports a subitizing-versus-estimation split. The mechanistic counting question is already studied in depth there.
- **Can Vision-Language Models Count? A Synthetic Benchmark and Analysis of Attention-Based Interventions** (arXiv:2511.17722, 2025). Controlled synthetic counting benchmark with attention interventions; similar one-variable-at-a-time spirit.
- **Do VLMs Perceive or Recall? Probing Visual Perception vs. Memory with Classic Visual Illusions** (arXiv:2601.22150, 2026). Introduces VI-Probe: graded counterfactual perturbations of classic illusions with matched controls; reports that persistence of illusory answers differs across model families. The evidence is behavioral. This motivates Act 2.

**Note**: the counting result replicates existing findings. The planned contribution, if any, is applying layer-wise probing and activation patching to the illusion setting, which I did not find done in the papers above.

## What's next: Act 2

The same question on classic visual illusions with counterfactual edits (image altered so the illusion no longer holds). The paper "Do VLMs Perceive or Recall? Probing Visual Perception vs. Memory with Classic Visual Illusions" reports persistent illusory answers but does not probe internal representations. 
We can try this as our next step: behavioral replication, layer-wise probing, then activation patching.
