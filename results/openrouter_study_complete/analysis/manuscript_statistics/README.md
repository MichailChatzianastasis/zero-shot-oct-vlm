# Manuscript statistics

## Scope and assumptions

These analyses use the saved predictions for 919 OCT B-scans. Each image
represents a different patient, so the patient and image are the same
statistical unit. Reference diagnoses were established by majority grading of
four ophthalmologists.

The confidence intervals quantify sampling uncertainty for this benchmark. They
do not measure API stochasticity, prompt sensitivity, model-version drift or
external validity.

## Statistical methods

- Accuracy and other binomial proportions: Wilson 95% confidence intervals.
- Macro-F1 and per-class F1: percentile 95% confidence intervals from 10,000
  paired, class-stratified bootstrap samples (seed 20260921).
- Model accuracy comparisons: paired differences with class-stratified
  bootstrap 95% confidence intervals and exact McNemar tests.
- Multiple comparisons: Holm correction across the three model pairs.

## Primary results

- Gemini-3-Flash: 815/919 correct; accuracy 88.7% (95% CI 86.5–90.6);
  macro-F1 0.886 (95% CI 0.866–0.906).
- GPT-5.4-mini: 556/919 correct; accuracy 60.5% (95% CI 57.3–63.6);
  macro-F1 0.598 (95% CI 0.568–0.627).
- Llama-3.2-11B-Vision: 159/919 correct; accuracy 17.3% (95% CI
  15.0–19.9); macro-F1 0.064 (95% CI 0.054–0.075).
- Gemini exceeded GPT by 28.2 percentage points (95% CI 25.0–31.3).
  Gemini alone was correct on 285 discordant cases and GPT alone on 26
  (exact McNemar P = 2.89 × 10^-56; Holm-adjusted P = 2.89 × 10^-56).
