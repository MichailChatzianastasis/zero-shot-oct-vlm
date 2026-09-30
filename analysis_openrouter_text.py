"""
Text-content analysis of the free-text rationales produced by each VLM.

Not just length — we look at:
  1. Clinical-term usage per predicted class (keyword concordance).
  2. Hedging / uncertainty vocabulary vs correctness.
  3. Lexical diversity (type-token ratio) per model.
  4. Boilerplate / prompt-copying score (n-gram overlap with the system
     prompt's per-class bullet lists) per model.
  5. Broad keyword overlap between a rationale and the reference class.

Keyword overlap is descriptive and does not establish that generated
rationales faithfully reveal the model's perceptual or causal process.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

CLASSES = ["CNV", "DRUSEN", "DME", "CSR", "MH", "NORMAL"]

MODELS = {
    "GPT-5.4-mini": "openai_gpt-5.4-mini_predictions.csv",
    "Gemini-3-Flash": "google_gemini-3-flash-preview_predictions.csv",
    "Llama-3.2-11B-Vision": "meta-llama_llama-3.2-11b-vision-instruct_predictions.csv",
}

# The two proprietary models we report on in the abstract; Llama is included
# only in the boilerplate / lexical-diversity comparison as a sanity baseline.
FOCUS_MODELS = ["Gemini-3-Flash", "GPT-5.4-mini"]

# Clinical-feature vocabulary organised per class.  Matches are case-insensitive
# substring hits so we cover plurals, compounds ("subretinal", "sub-retinal") etc.
CLASS_TERMS = {
    "CNV": [
        "choroidal neovascular", "neovascular membrane", "neovascularization",
        "subretinal hyperreflective", "shrm", "fibrovascular",
        "type 1 mnv", "type 2 mnv", "pigment epithelial detachment", "ped",
    ],
    "DRUSEN": [
        "drusen", "druse", "sub-rpe deposit", "sub rpe deposit",
        "rpe elevation", "bruch", "bruch's membrane", "reticular pseudodrusen",
    ],
    "DME": [
        "diabetic macular edema", "cystoid", "cystoid macular edema",
        "intraretinal cyst", "intraretinal fluid", "irf", "hard exudate",
        "retinal thickening", "dme",
    ],
    "CSR": [
        "central serous", "serous detachment", "neurosensory detachment",
        "dome-shaped", "dome shaped", "subretinal fluid", "srf",
        "csc", "csr",
    ],
    "MH": [
        "macular hole", "full-thickness", "full thickness",
        "foveal defect", "operculum", "lamellar hole",
    ],
    "NORMAL": [
        "normal", "healthy", "intact", "preserved", "no fluid",
        "foveal depression", "unremarkable",
    ],
}

HEDGE_TERMS = [
    "consistent with", "suggestive of", "appears", "appear to",
    "likely", "possibly", "possible", "may represent", "could represent",
    "suspicious for", "raising the possibility", "concerning for",
    "most likely", "probable", "probably",
]

DIFFERENTIAL_MARKERS = [
    "rather than", "as opposed to", "differential", "versus",
    "vs.", "vs ", "not consistent with", "no evidence of",
    "without signs of", "argues against", "less likely",
]

RESULTS_DIR = Path(__file__).resolve().parent / "results" / "openrouter_study_complete"
OUT_DIR = RESULTS_DIR / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z\-']+")


def load():
    dfs = {}
    for name, fname in MODELS.items():
        df = pd.read_csv(RESULTS_DIR / fname)
        df = df[df["predicted_label"].isin(CLASSES)].copy()
        df["explanation"] = df["explanation"].fillna("").astype(str)
        df["expl_lower"] = df["explanation"].str.lower()
        df["correct"] = df["true_label"] == df["predicted_label"]
        dfs[name] = df
    return dfs


# ---------------------------------------------------------------------------
# 1. Clinical-term usage: does the rationale contain terms matching the
#    predicted class?
# ---------------------------------------------------------------------------

def term_hit(text_lower: str, terms: list[str]) -> bool:
    return any(t in text_lower for t in terms)


def faithfulness_analysis(dfs):
    rows = []
    for name, df in dfs.items():
        for cls in CLASSES:
            sub = df[df["predicted_label"] == cls]
            if len(sub) == 0:
                continue
            hits = sub["expl_lower"].apply(lambda t: term_hit(t, CLASS_TERMS[cls]))
            rows.append({
                "model": name,
                "predicted_class": cls,
                "n_predictions": len(sub),
                "predicted_class_keyword_hits": int(hits.sum()),
                "keyword_concordance_rate": float(hits.mean()),
            })
    return pd.DataFrame(rows)


def true_vs_pred_mention(dfs):
    """Broad keyword overlap with the reference class on wrong predictions.

    Shared terms such as "subretinal fluid" are non-specific, so a hit does
    not imply that the model perceived the correct diagnosis.
    """
    rows = []
    for name, df in dfs.items():
        wrong = df[~df["correct"]]
        for cls in CLASSES:
            sub = wrong[wrong["true_label"] == cls]
            if len(sub) == 0:
                continue
            mentions_true = sub["expl_lower"].apply(lambda t: term_hit(t, CLASS_TERMS[cls]))
            rows.append({
                "model": name,
                "true_class": cls,
                "n_wrong": len(sub),
                "broad_reference_keyword_hits": int(mentions_true.sum()),
                "broad_reference_keyword_rate": float(mentions_true.mean()),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Hedging / uncertainty vocabulary vs correctness.
# ---------------------------------------------------------------------------

def hedging_analysis(dfs):
    rows = []
    for name, df in dfs.items():
        df = df.copy()
        df["n_hedges"] = df["expl_lower"].apply(lambda t: sum(t.count(h) for h in HEDGE_TERMS))
        df["n_diffs"] = df["expl_lower"].apply(lambda t: sum(t.count(d) for d in DIFFERENTIAL_MARKERS))
        c_hedges = df.loc[df["correct"], "n_hedges"]
        w_hedges = df.loc[~df["correct"], "n_hedges"]
        c_diff = df.loc[df["correct"], "n_diffs"]
        w_diff = df.loc[~df["correct"], "n_diffs"]
        stat_h, p_h = stats.mannwhitneyu(c_hedges, w_hedges, alternative="two-sided") if len(w_hedges) else (np.nan, np.nan)
        stat_d, p_d = stats.mannwhitneyu(c_diff, w_diff, alternative="two-sided") if len(w_diff) else (np.nan, np.nan)
        rows.append({
            "model": name,
            "mean_hedges_correct": c_hedges.mean(),
            "mean_hedges_incorrect": w_hedges.mean(),
            "hedges_U_p": p_h,
            "mean_differentials_correct": c_diff.mean(),
            "mean_differentials_incorrect": w_diff.mean(),
            "differentials_U_p": p_d,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Lexical diversity per model.
# ---------------------------------------------------------------------------

def lexical_diversity(dfs):
    rows = []
    for name, df in dfs.items():
        all_tokens = []
        per_row_ttr = []
        for t in df["expl_lower"]:
            toks = WORD_RE.findall(t)
            all_tokens.extend(toks)
            if toks:
                per_row_ttr.append(len(set(toks)) / len(toks))
        ttr_corpus = len(set(all_tokens)) / max(len(all_tokens), 1)
        rows.append({
            "model": name,
            "n_total_tokens": len(all_tokens),
            "n_unique_tokens": len(set(all_tokens)),
            "corpus_type_token_ratio": ttr_corpus,
            "mean_per_rationale_ttr": float(np.mean(per_row_ttr)) if per_row_ttr else 0.0,
            "mean_rationale_len_tokens": float(np.mean([len(WORD_RE.findall(t)) for t in df["expl_lower"]])),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. Boilerplate / prompt-copying: mean per-rationale frequency of the most
#    common shared 4-grams across all rationales of that model.  A high
#    share-of-top-4grams score = each rationale is largely drawn from the
#    same canned phrases, i.e. boilerplate.
# ---------------------------------------------------------------------------

def ngrams(tokens, n):
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def boilerplate_score(dfs, n=4, top_k=20):
    rows = []
    all_top_ngrams = {}
    for name, df in dfs.items():
        corpus_ngrams = Counter()
        per_doc_ngrams = []
        for t in df["expl_lower"]:
            toks = WORD_RE.findall(t)
            grams = ngrams(toks, n)
            per_doc_ngrams.append(grams)
            corpus_ngrams.update(grams)
        top = corpus_ngrams.most_common(top_k)
        all_top_ngrams[name] = top
        top_set = set(g for g, _ in top)
        coverage = []
        for grams in per_doc_ngrams:
            if not grams:
                coverage.append(0.0)
                continue
            hit = sum(1 for g in grams if g in top_set)
            coverage.append(hit / len(grams))
        rows.append({
            "model": name,
            "top20_4gram_coverage_mean": float(np.mean(coverage)),
            "top20_4gram_coverage_median": float(np.median(coverage)),
            "example_top_4grams": " | ".join(" ".join(g) for g, _ in top[:5]),
        })
    return pd.DataFrame(rows), all_top_ngrams


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_faithfulness(faith, out_path):
    faith_focus = faith[faith["model"].isin(FOCUS_MODELS)]
    pivot = faith_focus.pivot(
        index="predicted_class",
        columns="model",
        values="keyword_concordance_rate",
    ).reindex(CLASSES)
    pivot = pivot[FOCUS_MODELS]
    fig, ax = plt.subplots(figsize=(10, 5))
    pivot.plot(kind="bar", ax=ax, width=0.7,
               color=["#4285f4", "#10a37f"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Fraction of rationales containing\na predicted-class term")
    ax.set_xlabel("Predicted class")
    ax.set_title("Rationale keyword concordance with the predicted class",
                 fontweight="bold")
    ax.legend(title="Model")
    ax.set_xticklabels(CLASSES, rotation=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_boilerplate(boiler, out_path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    order = boiler.sort_values("top20_4gram_coverage_mean", ascending=False)
    bars = ax.bar(order["model"], order["top20_4gram_coverage_mean"],
                  color=["#f5a623", "#4285f4", "#10a37f"])
    ax.set_ylabel("Mean fraction of 4-grams drawn\nfrom the 20 most common 4-grams")
    ax.set_title("Boilerplate score — high = rationales recycle the same canned phrases",
                 fontweight="bold")
    for bar, v in zip(bars, order["top20_4gram_coverage_mean"].values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{v:.2f}", ha="center", fontsize=10)
    ax.set_ylim(0, max(0.35, order["top20_4gram_coverage_mean"].max() + 0.05))
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_hedging(hedge, dfs, out_path):
    hedge = hedge[hedge["model"].isin(FOCUS_MODELS)].set_index("model").loc[FOCUS_MODELS].reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    # Panel 1: mean hedge count per model, correct vs incorrect
    x = np.arange(len(hedge))
    w = 0.35
    axes[0].bar(x - w / 2, hedge["mean_hedges_correct"], w, label="correct", color="#4c9f70")
    axes[0].bar(x + w / 2, hedge["mean_hedges_incorrect"], w, label="incorrect", color="#c44e52")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(hedge["model"], rotation=0)
    axes[0].set_ylabel("Mean # hedging expressions per rationale")
    axes[0].set_title("Hedging vocabulary", fontweight="bold")
    axes[0].legend()
    for i, (p, yc, yw) in enumerate(zip(hedge["hedges_U_p"], hedge["mean_hedges_correct"], hedge["mean_hedges_incorrect"])):
        star = "**" if p < 0.01 else ("*" if p < 0.05 else "")
        if star:
            axes[0].text(i, max(yc, yw) + 0.05, star, ha="center", fontsize=14)
    # Panel 2: differentials
    axes[1].bar(x - w / 2, hedge["mean_differentials_correct"], w, label="correct", color="#4c9f70")
    axes[1].bar(x + w / 2, hedge["mean_differentials_incorrect"], w, label="incorrect", color="#c44e52")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(hedge["model"], rotation=0)
    axes[1].set_ylabel("Mean # differential-diagnosis markers per rationale")
    axes[1].set_title("Explicit differential-diagnosis reasoning", fontweight="bold")
    axes[1].legend()
    for i, p in enumerate(hedge["differentials_U_p"]):
        yc = hedge["mean_differentials_correct"].iloc[i]
        yw = hedge["mean_differentials_incorrect"].iloc[i]
        star = "**" if p < 0.01 else ("*" if p < 0.05 else "")
        if star:
            axes[1].text(i, max(yc, yw) + 0.01, star, ha="center", fontsize=14)
    fig.suptitle("Text-content signals of correctness  (** p<0.01, * p<0.05, Mann–Whitney U)",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_true_vs_pred(tvp, out_path):
    tvp = tvp[tvp["model"].isin(FOCUS_MODELS)]
    pivot = tvp.pivot(
        index="true_class",
        columns="model",
        values="broad_reference_keyword_rate",
    ).reindex(CLASSES)
    pivot = pivot[FOCUS_MODELS]
    fig, ax = plt.subplots(figsize=(10, 5))
    pivot.plot(kind="bar", ax=ax, width=0.7,
               color=["#4285f4", "#10a37f"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Fraction of wrong rationales containing\na broad reference-class keyword")
    ax.set_xlabel("True class (among misclassified samples)")
    ax.set_title("Broad reference-class keyword overlap on misclassified images",
                 fontweight="bold")
    ax.legend(title="Model")
    ax.set_xticklabels(CLASSES, rotation=0)
    # Highlight the CSR-on-Gemini bar specifically
    ax.annotate(
        "Broad CSR hit is mainly\nnon-specific subretinal fluid",
        xy=(3 - 0.18, 0.94), xytext=(3.5, 0.7),
        arrowprops=dict(arrowstyle="->", color="black"),
        fontsize=9, ha="left",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    dfs = load()

    faith = faithfulness_analysis(dfs)
    faith.to_csv(OUT_DIR / "rationale_faithfulness.csv", index=False)
    print("\n=== KEYWORD CONCORDANCE WITH THE PREDICTED CLASS ===")
    print(faith.to_string(index=False))

    tvp = true_vs_pred_mention(dfs)
    tvp.to_csv(OUT_DIR / "rationale_mentions_true_class_when_wrong.csv", index=False)
    print("\n=== BROAD REFERENCE-CLASS KEYWORD OVERLAP ON WRONG PREDICTIONS ===")
    print(tvp.to_string(index=False))

    hedge = hedging_analysis(dfs)
    hedge.to_csv(OUT_DIR / "hedging_vs_correctness.csv", index=False)
    print("\n=== HEDGING / DIFFERENTIALS vs CORRECTNESS ===")
    print(hedge.to_string(index=False))

    lex = lexical_diversity(dfs)
    lex.to_csv(OUT_DIR / "lexical_diversity.csv", index=False)
    print("\n=== LEXICAL DIVERSITY ===")
    print(lex.to_string(index=False))

    boiler, top_ngrams = boilerplate_score(dfs)
    boiler.to_csv(OUT_DIR / "boilerplate_score.csv", index=False)
    with open(OUT_DIR / "top_4grams_per_model.json", "w") as f:
        json.dump({k: [(" ".join(g), c) for g, c in v] for k, v in top_ngrams.items()}, f, indent=2)
    print("\n=== BOILERPLATE (top-20 4-gram coverage) ===")
    print(boiler.to_string(index=False))

    plot_faithfulness(faith, OUT_DIR / "rationale_faithfulness.png")
    plot_true_vs_pred(tvp, OUT_DIR / "rationale_true_class_mention_on_errors.png")
    plot_hedging(hedge, dfs, OUT_DIR / "hedging_vs_correctness.png")
    plot_boilerplate(boiler, OUT_DIR / "boilerplate_score.png")

    print(f"\nText-analysis artifacts written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
