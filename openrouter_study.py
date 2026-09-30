"""
Multi-model OCT classification study using OpenRouter API.

Evaluates multiple vision-language models on OCT image classification
(CNV vs DRUSEN vs NORMAL) through OpenRouter's unified API.
"""

import os
import json
import base64
import time
from io import BytesIO
from PIL import Image
import numpy as np
from pathlib import Path
from openai import OpenAI
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd
import argparse
from datetime import datetime
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]

CLASSES = ['CNV', 'DRUSEN', 'DME', 'CSR', 'MH', 'NORMAL']

FOLDER_TO_CLASS = {
    "1 CNV all": "CNV",
    "2 DRUSEN all": "DRUSEN",
    "3 DME all": "DME",
    "4 CSR all": "CSR",
    "5 MH all": "MH",
    "6 NORMAL all": "NORMAL",
}

# Models that don't support system messages — fold system prompt into user message
NO_SYSTEM_MSG_MODELS = {"google/gemma-3-27b-it:free"}

# Free-tier models need longer pauses to stay under rate limits
FREE_MODELS = {"google/gemma-3-27b-it:free"}

SYSTEM_PROMPT = (
    "You are a highly experienced retinal specialist and OCT (Optical Coherence Tomography) image analyst "
    "with extensive expertise in diagnosing retinal pathologies. "
    "OCT is a non-invasive imaging technique that provides cross-sectional views of the retina, "
    "allowing visualization of retinal layers and pathological changes. "
    "\n\nYour task is to analyze OCT images and classify them into one of six categories:\n\n"
    "1. CNV (Choroidal Neovascularization): Abnormal growth of new blood vessels from the choroid "
    "into the subretinal space. Key features:\n"
    "   - Hyperreflective material or fluid in the subretinal space\n"
    "   - Disruption or elevation of the retinal pigment epithelium (RPE) layer\n"
    "   - Subretinal or intraretinal fluid accumulation\n"
    "   - Presence of neovascular membrane (hyperreflective tissue)\n\n"
    "2. DRUSEN: Extracellular deposits between the RPE and Bruch's membrane. Key features:\n"
    "   - Small to medium-sized hyperreflective deposits beneath the RPE\n"
    "   - RPE elevation or irregularity overlying the deposits\n"
    "   - Well-defined, rounded or oval-shaped lesions\n"
    "   - RPE undulation but generally intact, no significant fluid\n\n"
    "3. DME (Diabetic Macular Edema): Fluid accumulation in the macula due to diabetic retinopathy. Key features:\n"
    "   - Intraretinal cystoid spaces (hyporeflective cavities within the retina)\n"
    "   - Diffuse retinal thickening\n"
    "   - Possible subretinal fluid\n"
    "   - Hard exudates (hyperreflective dots in outer retinal layers)\n"
    "   - Disruption of the normal foveal contour\n\n"
    "4. CSR (Central Serous Retinopathy): Serous detachment of the neurosensory retina. Key features:\n"
    "   - Well-defined dome-shaped elevation of the neurosensory retina\n"
    "   - Clear subretinal fluid (hyporeflective space between retina and RPE)\n"
    "   - RPE layer usually intact but may show focal irregularities or PED\n"
    "   - Retinal layers above the detachment are typically preserved\n"
    "   - No intraretinal cysts or significant retinal thickening\n\n"
    "5. MH (Macular Hole): Full-thickness defect in the neurosensory retina at the fovea. Key features:\n"
    "   - Full-thickness interruption of retinal tissue at the fovea\n"
    "   - Surrounding cuff of intraretinal fluid or cystoid changes\n"
    "   - Elevated edges of the hole\n"
    "   - Possible operculum (detached retinal tissue above the hole)\n"
    "   - Disruption of the photoreceptor layer\n\n"
    "6. NORMAL: Healthy retina with no pathological changes. Key features:\n"
    "   - Well-defined, continuous retinal layers with normal architecture\n"
    "   - Intact and smooth RPE layer\n"
    "   - No subretinal or intraretinal fluid\n"
    "   - Normal foveal depression\n"
    "   - Clear separation between retinal layers\n\n"
    "When analyzing an OCT image, carefully examine:\n"
    "- The integrity and continuity of retinal layers\n"
    "- The presence and location of any fluid (subretinal, intraretinal, or sub-RPE)\n"
    "- Hyperreflective or hyporeflective abnormalities\n"
    "- RPE layer morphology and continuity\n"
    "- Foveal contour and macular architecture\n"
    "- Overall retinal architecture\n\n"
    "Your prediction should be based only on the image provided. Base your assessment on the provided image, strictly on imaging features, do not make assumptions."
    "Ignore any text information, imaging metadata or image description."
    "You MUST respond with ONLY a valid JSON object with two fields:\n"
    '- "prediction": one of "CNV", "DRUSEN", "DME", "CSR", "MH", or "NORMAL"\n'
    '- "explanation": a brief clinical explanation of your reasoning\n\n'
    "Do not include any other text outside the JSON object."
)

USER_PROMPT = (
    "Analyze this OCT (Optical Coherence Tomography) image of the retina and classify it "
    "into one of the following categories: CNV, DRUSEN, DME, CSR, MH, or NORMAL.\n\n"
    "Respond with ONLY a JSON object like: "
    '{"prediction": "CNV", "explanation": "..."}'
)


def encode_image_to_base64(image_path):
    with Image.open(image_path) as img:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        buffered = BytesIO()
        img.save(buffered, format="JPEG", quality=85)
        return base64.b64encode(buffered.getvalue()).decode('utf-8')


def create_client():
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )


def get_prediction(client, model_id, image_path, max_retries=3):
    """Get classification prediction from a model via OpenRouter."""
    base64_image = encode_image_to_base64(image_path)

    if model_id in NO_SYSTEM_MSG_MODELS:
        combined_text = SYSTEM_PROMPT + "\n\n" + USER_PROMPT
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": combined_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                ],
            },
        ]
    else:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": USER_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                ],
            },
        ]

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=messages,
                max_tokens=1024,
                temperature=0.0,
            )

            content = response.choices[0].message.content.strip()
            cost = getattr(response, 'usage', None)
            token_info = {}
            if cost:
                token_info = {
                    'prompt_tokens': cost.prompt_tokens,
                    'completion_tokens': cost.completion_tokens,
                    'total_tokens': cost.total_tokens,
                }

            parsed = parse_response(content)
            return {
                'prediction': parsed['prediction'],
                'explanation': parsed['explanation'],
                'raw_response': content,
                **token_info,
            }

        except Exception as e:
            err_str = str(e)
            if attempt < max_retries - 1:
                is_rate_limit = "429" in err_str or "rate limit" in err_str.lower()
                base_wait = 30 if (is_rate_limit and model_id in FREE_MODELS) else 2 ** (attempt + 1)
                wait = base_wait * (attempt + 1)
                print(f"  Retry {attempt+1}/{max_retries} for {model_id}, waiting {wait}s ...")
                time.sleep(wait)
            else:
                print(f"  Failed after {max_retries} attempts: {e}")
                return {
                    'prediction': 'ERROR',
                    'explanation': err_str,
                    'raw_response': err_str,
                }


def parse_response(content):
    """Extract prediction and explanation from model response."""
    # Try to parse JSON directly
    cleaned = content.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()

    try:
        data = json.loads(cleaned)
        prediction = data.get("prediction", "").upper().strip()
        explanation = data.get("explanation", "")
        if prediction in CLASSES:
            return {"prediction": prediction, "explanation": explanation}
        # Try fuzzy match
        for cls in CLASSES:
            if cls in prediction:
                return {"prediction": cls, "explanation": explanation}
        return {"prediction": "UNKNOWN", "explanation": explanation}
    except json.JSONDecodeError:
        pass

    # Fallback: search for class name or partial matches in raw text
    upper = content.upper()
    for cls in CLASSES:
        if cls in upper:
            return {"prediction": cls, "explanation": content}

    # Handle truncated class names
    partial_map = {
        "DRUS": "DRUSEN", "DRUSE": "DRUSEN",
        "NOR": "NORMAL", "NORM": "NORMAL",
        "MACULAR HOLE": "MH", "MACULAR_HOLE": "MH",
        "DIABETIC MACULAR": "DME",
        "CENTRAL SEROUS": "CSR",
        "CHOROIDAL NEOVASC": "CNV",
    }
    for partial, full in partial_map.items():
        if partial in upper:
            return {"prediction": full, "explanation": content}

    return {"prediction": "UNKNOWN", "explanation": content}


def collect_samples(dataset_dir, samples_per_class=None, seed=42):
    """Collect image paths and labels from the external dataset structure."""
    samples = []
    for folder_name, class_name in FOLDER_TO_CLASS.items():
        class_dir = Path(dataset_dir) / folder_name
        if not class_dir.exists():
            print(f"Warning: {class_dir} not found, skipping")
            continue

        image_files = sorted(
            [f for f in class_dir.iterdir() if f.suffix.lower() in ('.jpg', '.jpeg', '.png')]
        )

        if samples_per_class is not None and samples_per_class < len(image_files):
            rng = np.random.RandomState(seed)
            indices = rng.choice(len(image_files), size=samples_per_class, replace=False)
            image_files = [image_files[i] for i in sorted(indices)]

        for img_path in image_files:
            samples.append((img_path, class_name))

    return samples


def evaluate_model(client, model_id, samples, results_dir):
    """Evaluate a single model on the given samples."""
    model_short = model_id.replace("/", "_").replace(":", "_")
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_id}")
    print(f"{'='*60}")

    predictions = []
    for img_path, true_label in tqdm(samples, desc=model_id):
        result = get_prediction(client, model_id, img_path)
        predictions.append({
            'image_path': str(img_path),
            'filename': img_path.name,
            'true_label': true_label,
            'predicted_label': result['prediction'],
            'explanation': result['explanation'],
            'raw_response': result.get('raw_response', ''),
            'prompt_tokens': result.get('prompt_tokens', 0),
            'completion_tokens': result.get('completion_tokens', 0),
            'total_tokens': result.get('total_tokens', 0),
        })
        delay = 5.0 if model_id in FREE_MODELS else 0.5
        time.sleep(delay)

    df = pd.DataFrame(predictions)
    df.to_csv(results_dir / f"{model_short}_predictions.csv", index=False)

    # Filter out errors/unknowns for metrics
    valid = df[df['predicted_label'].isin(CLASSES)]
    true_labels = valid['true_label'].tolist()
    pred_labels = valid['predicted_label'].tolist()

    if len(true_labels) == 0:
        print(f"  No valid predictions for {model_id}")
        return {'model': model_id, 'accuracy': 0, 'valid_predictions': 0, 'total_samples': len(samples), 'errors': len(samples), 'total_tokens': 0}

    acc = accuracy_score(true_labels, pred_labels)
    report = classification_report(true_labels, pred_labels, labels=CLASSES, zero_division=0)

    with open(results_dir / f"{model_short}_report.txt", "w") as f:
        f.write(f"Model: {model_id}\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Valid predictions: {len(valid)}/{len(samples)}\n\n")
        f.write(report)

    # Confusion matrix
    cm = confusion_matrix(true_labels, pred_labels, labels=CLASSES)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
    ax.set_title(f'Confusion Matrix: {model_id}')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    fig.tight_layout()
    fig.savefig(results_dir / f"{model_short}_confusion_matrix.png", dpi=150)
    plt.close(fig)

    print(f"  Accuracy: {acc:.4f} ({len(valid)}/{len(samples)} valid)")
    print(report)

    errors = len(samples) - len(valid)
    total_tokens = df['total_tokens'].sum()

    return {
        'model': model_id,
        'accuracy': acc,
        'valid_predictions': len(valid),
        'total_samples': len(samples),
        'errors': errors,
        'total_tokens': int(total_tokens),
    }


def create_comparison_report(all_results, results_dir):
    """Create a cross-model comparison."""
    summary_df = pd.DataFrame(all_results)
    summary_df = summary_df.sort_values('accuracy', ascending=False)
    summary_df.to_csv(results_dir / "model_comparison.csv", index=False)

    # Comparison bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    models = [r['model'].split('/')[-1] for r in all_results]
    accs = [r['accuracy'] for r in all_results]
    sorted_pairs = sorted(zip(models, accs), key=lambda x: x[1], reverse=True)
    models_sorted, accs_sorted = zip(*sorted_pairs) if sorted_pairs else ([], [])

    bars = ax.bar(range(len(models_sorted)), accs_sorted, color=sns.color_palette("viridis", len(models_sorted)))
    ax.set_xticks(range(len(models_sorted)))
    ax.set_xticklabels(models_sorted, rotation=45, ha='right')
    ax.set_ylabel('Accuracy')
    ax.set_title('OCT Classification Accuracy by Model')
    ax.set_ylim(0, 1.05)
    for bar, acc in zip(bars, accs_sorted):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{acc:.2%}', ha='center', va='bottom', fontsize=9)
    fig.tight_layout()
    fig.savefig(results_dir / "model_comparison.png", dpi=150)
    plt.close(fig)

    # Per-class accuracy per model (from saved predictions)
    per_class_data = []
    for result in all_results:
        model_short = result['model'].replace("/", "_").replace(":", "_")
        pred_file = results_dir / f"{model_short}_predictions.csv"
        if pred_file.exists():
            df = pd.read_csv(pred_file)
            valid = df[df['predicted_label'].isin(CLASSES)]
            for cls in CLASSES:
                cls_rows = valid[valid['true_label'] == cls]
                if len(cls_rows) > 0:
                    cls_acc = (cls_rows['predicted_label'] == cls).mean()
                else:
                    cls_acc = 0.0
                per_class_data.append({
                    'model': result['model'].split('/')[-1],
                    'class': cls,
                    'accuracy': cls_acc,
                })

    if per_class_data:
        pc_df = pd.DataFrame(per_class_data)
        fig, ax = plt.subplots(figsize=(14, 6))
        pc_pivot = pc_df.pivot(index='model', columns='class', values='accuracy')
        pc_pivot.plot(kind='bar', ax=ax, width=0.8)
        ax.set_ylabel('Per-class Accuracy')
        ax.set_title('Per-class Accuracy by Model')
        ax.set_ylim(0, 1.15)
        ax.legend(title='Class')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        fig.tight_layout()
        fig.savefig(results_dir / "per_class_comparison.png", dpi=150)
        plt.close(fig)

    # Text summary
    with open(results_dir / "study_summary.txt", "w") as f:
        f.write("=" * 70 + "\n")
        f.write("OCT Classification Study - Multi-Model Comparison\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Classes: {CLASSES}\n")
        f.write(f"Models evaluated: {len(all_results)}\n\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Model':<45} {'Accuracy':>10} {'Valid':>8} {'Total':>8}\n")
        f.write("-" * 70 + "\n")
        for r in sorted(all_results, key=lambda x: x['accuracy'], reverse=True):
            f.write(f"{r['model']:<45} {r['accuracy']:>10.4f} {r['valid_predictions']:>8} {r['total_samples']:>8}\n")
        f.write("-" * 70 + "\n")

    print("\n" + "=" * 70)
    print("STUDY SUMMARY")
    print("=" * 70)
    print(f"\n{'Model':<45} {'Accuracy':>10}")
    print("-" * 55)
    for r in sorted(all_results, key=lambda x: x['accuracy'], reverse=True):
        print(f"{r['model']:<45} {r['accuracy']:>10.2%}")
    print()


def main():
    DEFAULT_MODELS = [
        "openai/gpt-5.4-mini",
        "google/gemini-3-flash-preview",
        #"google/gemma-3-27b-it:free",
        "meta-llama/llama-3.2-11b-vision-instruct",
    ]
    models_list = "\n".join(f"  - {m}" for m in DEFAULT_MODELS)
    parser = argparse.ArgumentParser(
        description='Multi-model OCT classification study via OpenRouter',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Default models (used when --models is omitted):\n{models_list}",
    )
    parser.add_argument('--dataset_dir', type=str,
                        default="datasets/updated_dataset",
                        help='Path to the dataset directory')
    parser.add_argument('--samples_per_class', type=int, default=None,
                        help='Number of samples per class (None = all)')
    parser.add_argument('--models', nargs='+', default=DEFAULT_MODELS, metavar='MODEL',
                        help='One or more OpenRouter model IDs to evaluate. '
                             'Any valid OpenRouter model ID works (e.g. openai/gpt-4o-mini).')
    parser.add_argument('--output_dir', type=str, default='results/openrouter_study_complete',
                        help='Output directory for results')
    args = parser.parse_args()

    results_dir = Path(args.output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    models_to_eval = args.models

    print(f"Dataset: {args.dataset_dir}")
    print(f"Models: {models_to_eval}")
    print(f"Samples per class: {args.samples_per_class or 'all'}")

    samples = collect_samples(args.dataset_dir, args.samples_per_class)
    print(f"\nTotal samples: {len(samples)}")
    for cls in CLASSES:
        count = sum(1 for _, c in samples if c == cls)
        print(f"  {cls}: {count}")

    client = create_client()

    all_results = []
    for model_id in models_to_eval:
        result = evaluate_model(client, model_id, samples, results_dir)
        all_results.append(result)

        # Save intermediate results
        pd.DataFrame(all_results).to_csv(results_dir / "model_comparison.csv", index=False)

    create_comparison_report(all_results, results_dir)
    print(f"\nAll results saved to: {results_dir}")


if __name__ == "__main__":
    main()
