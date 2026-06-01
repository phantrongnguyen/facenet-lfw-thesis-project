from datetime import datetime
from pathlib import Path
import json


def create_report_run(reports_dir):
    base_dir = Path(reports_dir)
    runs_dir = base_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    latest_file = base_dir / "latest_run.txt"
    latest_file.write_text(run_id + "\n", encoding="utf-8")

    return {
        "base_dir": base_dir,
        "runs_dir": runs_dir,
        "run_id": run_id,
        "run_dir": run_dir,
    }


def build_results_dataframe(results, thresholds):
    import pandas as pd

    rows = []

    for model, metrics in results.items():
        acc, far, frr, inference_time = metrics

        rows.append(
            {
                "model": model,
                "threshold": thresholds.get(model),
                "accuracy": acc,
                "far": far,
                "frr": frr,
                "avg_inference_time_sec": inference_time,
            }
        )

    return pd.DataFrame(rows)


def save_report_files(report_context, results_df, dataset_path, pairs_file, total_pairs):
    run_dir = report_context["run_dir"]

    csv_path = run_dir / "metrics_summary.csv"
    json_path = run_dir / "metrics_summary.json"
    markdown_path = run_dir / "summary.md"
    metadata_path = run_dir / "metadata.json"

    results_df.to_csv(csv_path, index=False)

    records = results_df.to_dict(orient="records")
    json_path.write_text(json.dumps(records, indent=2), encoding="utf-8")

    generated_at = datetime.now().isoformat(timespec="seconds")
    lines = [
        "# Face Recognition Evaluation Summary",
        "",
        f"- Generated at: {generated_at}",
        f"- Dataset path: {dataset_path}",
        f"- Pairs file: {pairs_file}",
        f"- Total evaluated pairs: {total_pairs}",
        f"- Number of models: {len(results_df)}",
        "",
        "## Metrics by model",
        "",
        "| model | threshold | accuracy | far | frr | avg_inference_time_sec |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for row in records:
        lines.append(
            f"| {row['model']} | {row['threshold']:.4f} | {row['accuracy']:.6f} | "
            f"{row['far']:.6f} | {row['frr']:.6f} | {row['avg_inference_time_sec']:.6f} |"
        )

    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    metadata = {
        "generated_at": generated_at,
        "dataset_path": dataset_path,
        "pairs_file": pairs_file,
        "total_pairs": total_pairs,
        "run_id": report_context["run_id"],
        "report_folder": str(run_dir),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "csv": str(csv_path),
        "json": str(json_path),
        "markdown": str(markdown_path),
        "metadata": str(metadata_path),
    }
