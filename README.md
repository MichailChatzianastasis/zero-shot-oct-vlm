# Zero-shot retinal OCT classification with general-purpose vision–language models

Code, model outputs and statistical analyses for the study *"Zero-shot classification of retinal optical coherence tomography by general-purpose vision-language models: diagnostic accuracy and explanation errors"* (manuscript in preparation).

Three general-purpose vision–language models (VLMs) were given the same expert-written prompt and asked to classify 919 macular OCT B-scans (one per patient, three Greek centres) into six classes: choroidal neovascularisation (CNV), drusen, diabetic macular oedema (DME), central serous chorioretinopathy (CSR), macular hole (MH) and normal retina. No fine-tuning or example images were used.

| Model (OpenRouter ID) | Correct | Accuracy (95% CI) | Macro-F1 (95% CI) |
|---|---:|---:|---:|
| `google/gemini-3-flash-preview` | 815/919 | 88.7% (86.5–90.6) | 0.886 (0.866–0.906) |
| `openai/gpt-5.4-mini` | 556/919 | 60.5% (57.3–63.6) | 0.598 (0.568–0.627) |
| `meta-llama/llama-3.2-11b-vision-instruct` | 159/919 | 17.3% (15.0–19.9) | 0.064 (0.054–0.075) |

## Data availability

The clinical OCT images are **not** included. They come from institutional archives and cannot be shared for patient-privacy and ethical reasons. This repository contains everything derived from them that is needed to reproduce the reported statistics: each model's prediction, free-text explanation, raw response and token usage for every image, identified only by an anonymised file code (for example `EX1 CNV 12.jpg`).

## Repository contents

```
openrouter_study.py                 # evaluation: sends each image + prompt to each model via OpenRouter
analysis_manuscript_statistics.py   # manuscript statistics: Wilson and bootstrap CIs, McNemar/Holm, keyword sensitivity
analysis_openrouter_study.py        # per-class metrics, agreement, summary figures
analysis_openrouter_text.py         # exploratory analysis of the models' explanations
make_figure2.py                     # manuscript Figure 2: row-normalised confusion matrices
requirements.txt
results/openrouter_study_complete/
├── *_predictions.csv               # one row per image: reference label, prediction, explanation, raw response, tokens
├── *_report.txt                    # per-model classification report
├── *_confusion_matrix.png          # per-model confusion matrix
└── analysis/
    ├── manuscript_statistics/      # all numbers reported in the manuscript (see its README)
    └── ...                         # per-class metrics, agreement and explanation analyses
```

The full system and user prompts are defined at the top of `openrouter_study.py` (`SYSTEM_PROMPT`, `USER_PROMPT`).

## Reproducing the analyses

The analyses only need the saved prediction files, so they run without the images or an API key.

```bash
pip install -r requirements.txt
python analysis_manuscript_statistics.py
python analysis_openrouter_study.py
python analysis_openrouter_text.py
```

Outputs are written to `results/openrouter_study_complete/analysis/`.

## Running the evaluation on your own images

`openrouter_study.py` expects one sub-folder per class with exactly these names: `1 CNV all`, `2 DRUSEN all`, `3 DME all`, `4 CSR all`, `5 MH all` and `6 NORMAL all` (JPEG or PNG files).

```bash
cp .env.example .env              # then add your OpenRouter API key
python openrouter_study.py --dataset_dir path/to/images --output_dir results/my_run
python openrouter_study.py --dataset_dir path/to/images --models anthropic/claude-3.5-sonnet
```

Generation settings: temperature 0, maximum 1,024 output tokens, JSON-formatted response with a diagnosis and a short explanation, up to three retries per failed call. Hosted models can change without notice, so re-running the evaluation may not reproduce the saved predictions exactly.
