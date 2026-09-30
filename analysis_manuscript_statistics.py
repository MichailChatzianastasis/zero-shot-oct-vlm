"""Manuscript-focused statistics for the OCT VLM benchmark.

Adds confidence intervals, paired model comparisons, confusion-rate
uncertainty, and a sensitivity analysis of rationale-keyword claims.
All resampling is paired and stratified by reference diagnosis.
"""

from itertools import combinations
import json
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd
from scipy.stats import binomtest


CLASSES = ["CNV", "DRUSEN", "DME", "CSR", "MH", "NORMAL"]

MODELS = {
    "Gemini-3-Flash": "google_gemini-3-flash-preview_predictions.csv",
    "GPT-5.4-mini": "openai_gpt-5.4-mini_predictions.csv",
    "Llama-3.2-11B-Vision": "meta-llama_llama-3.2-11b-vision-instruct_predictions.csv",
}

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results" / "openrouter_study_complete"
OUT_DIR = RESULTS_DIR / "analysis" / "manuscript_statistics"

N_BOOTSTRAP = 10_000
RANDOM_SEED = 20260921
CONFIDENCE = 0.95

# Broad vocabulary used in the submitted-abstract analysis. Several terms,
# such as "subretinal fluid", are shared by more than one diagnosis.
BROAD_CLASS_TERMS = {
    "CNV": [
        "choroidal neovascular",
        "neovascular membrane",
        "neovascularization",
        "subretinal hyperreflective",
        "shrm",
        "fibrovascular",
        "type 1 mnv",
        "type 2 mnv",
        "pigment epithelial detachment",
        "ped",
    ],
    "DRUSEN": [
        "drusen",
        "druse",
        "sub-rpe deposit",
        "sub rpe deposit",
        "rpe elevation",
        "bruch",
        "bruch's membrane",
        "reticular pseudodrusen",
    ],
    "DME": [
        "diabetic macular edema",
        "diabetic macular oedema",
        "cystoid",
        "cystoid macular edema",
        "cystoid macular oedema",
        "intraretinal cyst",
        "intraretinal fluid",
        "irf",
        "hard exudate",
        "retinal thickening",
        "dme",
    ],
    "CSR": [
        "central serous",
        "serous detachment",
        "neurosensory detachment",
        "dome-shaped",
        "dome shaped",
        "subretinal fluid",
        "srf",
        "csc",
        "csr",
    ],
    "MH": [
        "macular hole",
        "full-thickness",
        "full thickness",
        "foveal defect",
        "operculum",
        "lamellar hole",
    ],
    "NORMAL": [
        "normal",
        "healthy",
        "intact",
        "preserved",
        "no fluid",
        "foveal depression",
        "unremarkable",
    ],
}

# Restricted vocabulary excludes generic features shared by competing
# diagnoses. It is a sensitivity analysis, not a validation of rationale
# faithfulness.
SPECIFIC_CLASS_TERMS = {
    "CNV": [
        "choroidal neovascular",
        "neovascular membrane",
        "neovascularization",
        "shrm",
        "fibrovascular",
        "type 1 mnv",
        "type 2 mnv",
    ],
    "DRUSEN": ["drusen", "druse", "reticular pseudodrusen"],
    "DME": [
        "diabetic macular edema",
        "diabetic macular oedema",
        "cystoid macular edema",
        "cystoid macular oedema",
        "hard exudate",
        "dme",
    ],
    "CSR": [
        "central serous",
        "serous detachment",
        "neurosensory detachment",
        "dome-shaped",
        "dome shaped",
        "csc",
        "csr",
    ],
    "MH": [
        "macular hole",
        "full-thickness",
        "full thickness",
        "foveal defect",
        "operculum",
        "lamellar hole",
    ],
    "NORMAL": ["normal", "healthy", "unremarkable"],
}


def wilson_interval(successes, total, confidence=CONFIDENCE):
    """Wilson score interval for a binomial proportion."""
    if total == 0:
        return np.nan, np.nan
    z = NormalDist().inv_cdf(1 - (1 - confidence) / 2)
    proportion = successes / total
    denominator = 1 + z**2 / total
    centre = (proportion + z**2 / (2 * total)) / denominator
    half_width = (
        z
        * np.sqrt(proportion * (1 - proportion) / total + z**2 / (4 * total**2))
        / denominator
    )
    return max(0.0, centre - half_width), min(1.0, centre + half_width)


def percentile_interval(values, confidence=CONFIDENCE):
    """Two-sided percentile bootstrap interval."""
    alpha = (1 - confidence) / 2
    return tuple(np.quantile(values, [alpha, 1 - alpha]))


def divide(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def load_aligned_predictions():
    """Load predictions in a common sample order."""
    dataframes = {
        model: pd.read_csv(RESULTS_DIR / filename)
        for model, filename in MODELS.items()
    }
    first_model = next(iter(MODELS))
    aligned = dataframes[first_model][
        ["image_path", "filename", "true_label"]
    ].copy()

    for model, dataframe in dataframes.items():
        predictions = dataframe.set_index("image_path")["predicted_label"]
        aligned[model] = predictions.loc[aligned["image_path"]].to_numpy()

    return dataframes, aligned


def make_stratified_bootstrap(reference_codes):
    """Create one paired bootstrap index matrix shared by every model."""
    rng = np.random.default_rng(RANDOM_SEED)
    indices = np.empty(
        (N_BOOTSTRAP, len(reference_codes)),
        dtype=np.int32,
    )
    start = 0
    for class_code in range(len(CLASSES)):
        class_indices = np.flatnonzero(reference_codes == class_code)
        stop = start + len(class_indices)
        indices[:, start:stop] = rng.choice(
            class_indices,
            size=(N_BOOTSTRAP, len(class_indices)),
            replace=True,
        )
        start = stop
    return indices


def counts_for_class(reference_codes, prediction_codes, class_code):
    true_class = reference_codes == class_code
    predicted_class = prediction_codes == class_code
    tp = int(np.sum(true_class & predicted_class))
    fn = int(np.sum(true_class & ~predicted_class))
    fp = int(np.sum(~true_class & predicted_class))
    tn = int(np.sum(~true_class & ~predicted_class))
    return tp, fn, fp, tn


def observed_class_metrics(reference_codes, prediction_codes):
    rows = []
    for class_code, class_name in enumerate(CLASSES):
        tp, fn, fp, tn = counts_for_class(
            reference_codes,
            prediction_codes,
            class_code,
        )
        sensitivity = divide(tp, tp + fn)
        specificity = divide(tn, tn + fp)
        precision = divide(tp, tp + fp)
        f1 = divide(2 * tp, 2 * tp + fp + fn)
        rows.append(
            {
                "class": class_name,
                "tp": tp,
                "fn": fn,
                "fp": fp,
                "tn": tn,
                "sensitivity": sensitivity,
                "specificity": specificity,
                "precision": precision,
                "f1": f1,
            }
        )
    return rows


def bootstrap_model_metrics(
    reference_codes,
    prediction_codes,
    bootstrap_indices,
):
    """Return paired bootstrap distributions for one model."""
    reference_bootstrap = reference_codes[bootstrap_indices]
    prediction_bootstrap = prediction_codes[bootstrap_indices]
    accuracy = np.mean(
        reference_bootstrap == prediction_bootstrap,
        axis=1,
    )

    class_f1 = np.empty((N_BOOTSTRAP, len(CLASSES)))
    for class_code in range(len(CLASSES)):
        true_class = reference_bootstrap == class_code
        predicted_class = prediction_bootstrap == class_code
        tp = np.sum(true_class & predicted_class, axis=1)
        fn = np.sum(true_class & ~predicted_class, axis=1)
        fp = np.sum(~true_class & predicted_class, axis=1)
        denominator = 2 * tp + fp + fn
        class_f1[:, class_code] = np.divide(
            2 * tp,
            denominator,
            out=np.zeros_like(tp, dtype=float),
            where=denominator != 0,
        )

    return {
        "accuracy": accuracy,
        "class_f1": class_f1,
        "macro_f1": np.mean(class_f1, axis=1),
    }


def overall_and_class_statistics(
    reference_codes,
    prediction_codes_by_model,
    bootstrap_indices,
):
    overall_rows = []
    class_rows = []
    bootstrap_cache = {}

    for model, prediction_codes in prediction_codes_by_model.items():
        class_metrics = observed_class_metrics(
            reference_codes,
            prediction_codes,
        )
        bootstrap = bootstrap_model_metrics(
            reference_codes,
            prediction_codes,
            bootstrap_indices,
        )
        bootstrap_cache[model] = bootstrap

        n_correct = int(np.sum(reference_codes == prediction_codes))
        accuracy = n_correct / len(reference_codes)
        accuracy_low, accuracy_high = wilson_interval(
            n_correct,
            len(reference_codes),
        )
        macro_f1 = float(np.mean([row["f1"] for row in class_metrics]))
        macro_low, macro_high = percentile_interval(bootstrap["macro_f1"])
        overall_rows.append(
            {
                "model": model,
                "n": len(reference_codes),
                "n_correct": n_correct,
                "accuracy": accuracy,
                "accuracy_ci_low": accuracy_low,
                "accuracy_ci_high": accuracy_high,
                "macro_f1": macro_f1,
                "macro_f1_ci_low": macro_low,
                "macro_f1_ci_high": macro_high,
            }
        )

        for class_code, row in enumerate(class_metrics):
            sensitivity_low, sensitivity_high = wilson_interval(
                row["tp"],
                row["tp"] + row["fn"],
            )
            specificity_low, specificity_high = wilson_interval(
                row["tn"],
                row["tn"] + row["fp"],
            )
            precision_low, precision_high = wilson_interval(
                row["tp"],
                row["tp"] + row["fp"],
            )
            f1_low, f1_high = percentile_interval(
                bootstrap["class_f1"][:, class_code]
            )
            class_rows.append(
                {
                    "model": model,
                    **row,
                    "sensitivity_ci_low": sensitivity_low,
                    "sensitivity_ci_high": sensitivity_high,
                    "specificity_ci_low": specificity_low,
                    "specificity_ci_high": specificity_high,
                    "precision_ci_low": precision_low,
                    "precision_ci_high": precision_high,
                    "f1_ci_low": f1_low,
                    "f1_ci_high": f1_high,
                }
            )

    return (
        pd.DataFrame(overall_rows),
        pd.DataFrame(class_rows),
        bootstrap_cache,
    )


def holm_adjust(p_values):
    """Holm family-wise error correction."""
    p_values = np.asarray(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values))
    running_maximum = 0.0
    for rank, index in enumerate(order):
        candidate = (len(p_values) - rank) * p_values[index]
        running_maximum = max(running_maximum, candidate)
        adjusted[index] = min(running_maximum, 1.0)
    return adjusted


def paired_model_statistics(
    reference_codes,
    prediction_codes_by_model,
    bootstrap_cache,
):
    rows = []
    for model_a, model_b in combinations(MODELS, 2):
        correct_a = prediction_codes_by_model[model_a] == reference_codes
        correct_b = prediction_codes_by_model[model_b] == reference_codes
        a_only = int(np.sum(correct_a & ~correct_b))
        b_only = int(np.sum(~correct_a & correct_b))
        difference = float(np.mean(correct_a) - np.mean(correct_b))
        bootstrap_difference = (
            bootstrap_cache[model_a]["accuracy"]
            - bootstrap_cache[model_b]["accuracy"]
        )
        difference_low, difference_high = percentile_interval(
            bootstrap_difference
        )
        p_value = binomtest(
            a_only,
            a_only + b_only,
            p=0.5,
            alternative="two-sided",
        ).pvalue
        rows.append(
            {
                "model_a": model_a,
                "model_b": model_b,
                "accuracy_a": float(np.mean(correct_a)),
                "accuracy_b": float(np.mean(correct_b)),
                "accuracy_difference_a_minus_b": difference,
                "difference_ci_low": difference_low,
                "difference_ci_high": difference_high,
                "a_correct_b_wrong": a_only,
                "a_wrong_b_correct": b_only,
                "mcnemar_exact_p": p_value,
            }
        )

    dataframe = pd.DataFrame(rows)
    dataframe["mcnemar_holm_p"] = holm_adjust(
        dataframe["mcnemar_exact_p"].to_numpy()
    )
    return dataframe


def confusion_statistics(aligned):
    rows = []
    for model in MODELS:
        for true_class in CLASSES:
            class_rows = aligned[aligned["true_label"] == true_class]
            for predicted_class in CLASSES:
                count = int(
                    np.sum(class_rows[model] == predicted_class)
                )
                rate = count / len(class_rows)
                low, high = wilson_interval(count, len(class_rows))
                rows.append(
                    {
                        "model": model,
                        "true_class": true_class,
                        "predicted_class": predicted_class,
                        "count": count,
                        "true_class_n": len(class_rows),
                        "row_rate": rate,
                        "row_rate_ci_low": low,
                        "row_rate_ci_high": high,
                    }
                )
    return pd.DataFrame(rows)


def two_model_agreement_statistics(aligned):
    model_a = "Gemini-3-Flash"
    model_b = "GPT-5.4-mini"
    agree = aligned[model_a] == aligned[model_b]
    disagree = ~agree
    correct_a = aligned[model_a] == aligned["true_label"]
    correct_b = aligned[model_b] == aligned["true_label"]

    definitions = [
        ("agreement_rate", int(np.sum(agree)), len(aligned)),
        (
            "accuracy_when_models_agree",
            int(np.sum(correct_a & agree)),
            int(np.sum(agree)),
        ),
        (
            "gemini_accuracy_when_models_disagree",
            int(np.sum(correct_a & disagree)),
            int(np.sum(disagree)),
        ),
        (
            "gpt_accuracy_when_models_disagree",
            int(np.sum(correct_b & disagree)),
            int(np.sum(disagree)),
        ),
    ]
    rows = []
    for metric, successes, total in definitions:
        low, high = wilson_interval(successes, total)
        rows.append(
            {
                "metric": metric,
                "successes": successes,
                "n": total,
                "estimate": successes / total,
                "ci_low": low,
                "ci_high": high,
            }
        )
    return pd.DataFrame(rows)


def term_hit(text, terms):
    return any(term in text for term in terms)


def rationale_keyword_statistics(dataframes):
    rows = []
    for model, dataframe in dataframes.items():
        errors = dataframe[
            dataframe["predicted_label"] != dataframe["true_label"]
        ].copy()
        errors["explanation_lower"] = (
            errors["explanation"].fillna("").str.lower()
        )

        grouping_columns = [
            ("true_class", ["true_label"]),
            ("error_pair", ["true_label", "predicted_label"]),
        ]
        for grouping, columns in grouping_columns:
            grouper = columns[0] if len(columns) == 1 else columns
            for keys, group in errors.groupby(grouper):
                if isinstance(keys, str):
                    true_class = keys
                    predicted_class = "ANY_WRONG_CLASS"
                else:
                    true_class, predicted_class = keys

                broad_hits = group["explanation_lower"].apply(
                    lambda text: term_hit(
                        text,
                        BROAD_CLASS_TERMS[true_class],
                    )
                )
                specific_hits = group["explanation_lower"].apply(
                    lambda text: term_hit(
                        text,
                        SPECIFIC_CLASS_TERMS[true_class],
                    )
                )
                broad_low, broad_high = wilson_interval(
                    int(broad_hits.sum()),
                    len(group),
                )
                specific_low, specific_high = wilson_interval(
                    int(specific_hits.sum()),
                    len(group),
                )
                rows.append(
                    {
                        "model": model,
                        "grouping": grouping,
                        "true_class": true_class,
                        "predicted_class": predicted_class,
                        "n_errors": len(group),
                        "broad_keyword_hits": int(broad_hits.sum()),
                        "broad_keyword_rate": float(broad_hits.mean()),
                        "broad_keyword_ci_low": broad_low,
                        "broad_keyword_ci_high": broad_high,
                        "specific_keyword_hits": int(
                            specific_hits.sum()
                        ),
                        "specific_keyword_rate": float(
                            specific_hits.mean()
                        ),
                        "specific_keyword_ci_low": specific_low,
                        "specific_keyword_ci_high": specific_high,
                    }
                )
    return pd.DataFrame(rows)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dataframes, aligned = load_aligned_predictions()
    class_to_code = {
        class_name: class_code
        for class_code, class_name in enumerate(CLASSES)
    }
    reference_codes = aligned["true_label"].map(class_to_code).to_numpy()
    prediction_codes_by_model = {
        model: aligned[model].map(class_to_code).to_numpy()
        for model in MODELS
    }
    bootstrap_indices = make_stratified_bootstrap(reference_codes)

    overall, per_class, bootstrap_cache = overall_and_class_statistics(
        reference_codes,
        prediction_codes_by_model,
        bootstrap_indices,
    )
    paired = paired_model_statistics(
        reference_codes,
        prediction_codes_by_model,
        bootstrap_cache,
    )
    confusion = confusion_statistics(aligned)
    agreement = two_model_agreement_statistics(aligned)
    rationale = rationale_keyword_statistics(dataframes)

    overall.to_csv(OUT_DIR / "overall_metrics_with_ci.csv", index=False)
    per_class.to_csv(OUT_DIR / "per_class_metrics_with_ci.csv", index=False)
    paired.to_csv(OUT_DIR / "paired_model_comparisons.csv", index=False)
    confusion.to_csv(OUT_DIR / "confusion_rates_with_ci.csv", index=False)
    agreement.to_csv(
        OUT_DIR / "two_model_agreement_with_ci.csv",
        index=False,
    )
    rationale.to_csv(
        OUT_DIR / "rationale_keyword_sensitivity.csv",
        index=False,
    )

    manifest = {
        "confidence_level": CONFIDENCE,
        "accuracy_and_proportion_intervals": "Wilson score",
        "f1_and_accuracy_difference_intervals": (
            "paired stratified percentile bootstrap"
        ),
        "bootstrap_replicates": N_BOOTSTRAP,
        "bootstrap_seed": RANDOM_SEED,
        "paired_test": "exact McNemar test",
        "multiple_testing": "Holm correction across three model pairs",
        "statistical_unit": (
            "one image per independent patient, as confirmed by study team"
        ),
        "important_limitation": (
            "intervals quantify benchmark sampling uncertainty, not API "
            "stochasticity or temporal model-version variability"
        ),
    }
    with open(OUT_DIR / "analysis_manifest.json", "w") as file:
        json.dump(manifest, file, indent=2)

    print(overall.to_string(index=False))
    print("\nPaired model comparisons")
    print(paired.to_string(index=False))
    print(f"\nOutputs written to {OUT_DIR}")


if __name__ == "__main__":
    main()
