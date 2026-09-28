# VLM Probing

Controlled stress sweeps (density, occlusion) as a first step toward extreme cases for vision-language models, paired with layer-wise probing of internal representations.

**Status:** Act 1 (counting / occlusion): The preliminary stage where I read through similar papers and replicated is done.

## Motivation

Standard benchmarks give a single accuracy number. A more informative approach is to construct extreme, tightly controlled visual cases (one variable changes at a time) and observe *how* models fail. This repo asks a mechanistic version of that question:

> When a VLM gives a wrong answer on an extreme case, is the correct information absent from its internal representations, or present but not reflected in the output?

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

## Limitations

- Small sample (41 images)
- Hidden states are mean-pooled over all tokens (image and prompt), not visual tokens only.
- The layer where decodability peaks changed between a 19-image and a 41-image run for Qwen, so the claim is "decodable at every layer," not a specific layer.
- Probes are linear regressions on a dataset where count correlates with other image statistics. The size-varied control is supported by model behavior only; a regression on those 5 images was overfit.
- The alpha = 0 occlusion condition already has circles touching, so the fully separated and partially separated conditions were added to get a real baseline.

## What's next: Act 2

The same question on classic visual illusions with counterfactual edits (image altered so the illusion no longer holds). The paper "Do VLMs Perceive or Recall? Probing Visual Perception vs. Memory with Classic Visual Illusions" reports persistent illusory answers but does not probe internal representations. 
We can try this as our next step: behavioral replication, layer-wise probing, then activation patching.
