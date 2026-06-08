import csv
from pathlib import Path
from collections import defaultdict


def format_cer(value):
    """
    Format CER values for printing summaries.
    Keeps compatibility with summary.py.
    """

    if value is None:
        return "n/a"

    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def collect_fieldnames(rows):
    """
    Collect all CSV columns from all rows.

    This allows normal result rows and comparison rows
    to be stored in the same CSV file.
    """

    fieldnames = []

    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    return fieldnames


def write_results_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        return

    fieldnames = collect_fieldnames(rows)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )

        writer.writeheader()
        writer.writerows(rows)


def append_result_and_save(path, rows, row):
    rows.append(row)
    write_results_csv(path, rows)


# Printing helpers

def print_random_writer_mapping(random_writer_mapping):
    print("=" * 80)
    print("Random writer mapping")
    print("=" * 80)

    for target_writer_id, source_writer_id in random_writer_mapping.items():
        print(f"Target writer {target_writer_id} -> training writer {source_writer_id}")

    print("=" * 80)


def compute_mean_cer(rows, cer_name="cer"):
    if len(rows) == 0:
        return None

    return sum(row[cer_name] for row in rows) / len(rows)


def compute_mean_by_key(rows, key_name, cer_name="cer"):
    values_by_key = defaultdict(list)

    for row in rows:
        key = row[key_name]
        value = row[cer_name]
        values_by_key[key].append(value)

    mean_by_key = {}

    for key, values in values_by_key.items():
        mean_by_key[key] = sum(values) / len(values)

    return mean_by_key


def print_baseline_summary(baseline_results):
    baseline_mean = compute_mean_cer(baseline_results)

    print("\n" + "=" * 80)
    print("Evaluation baseline summary")
    print("=" * 80)
    print(f"Writers evaluated: {len(baseline_results)}")
    print(f"Baseline mean CER: {format_cer(baseline_mean)}")
    print("=" * 80)


def print_progressive_mean_summary(actual_results, random_results):
    actual_mean = compute_mean_by_key(
        rows=actual_results,
        key_name="sample_size",
    )

    random_mean = compute_mean_by_key(
        rows=random_results,
        key_name="target_sample_size",
    )

    all_sample_sizes = sorted(set(actual_mean.keys()) | set(random_mean.keys()))

    print("\n" + "=" * 80)
    print("Progressive mean CER summary")
    print("=" * 80)
    print("sample_size\tactual_writer_mean_cer\trandom_writer_mean_cer\tdifference")

    for sample_size in all_sample_sizes:
        actual_cer = actual_mean.get(sample_size)
        random_cer = random_mean.get(sample_size)

        if actual_cer is None or random_cer is None:
            difference = None
        else:
            difference = random_cer - actual_cer

        print(
            f"{sample_size}\t"
            f"{format_cer(actual_cer)}\t"
            f"{format_cer(random_cer)}\t"
            f"{format_cer(difference)}"
        )

    print("=" * 80)


def print_final_locations(baseline_csv, actual_csv, random_csv):
    print("\n" + "=" * 80)
    print("Saved result files")
    print("=" * 80)
    print(f"Evaluation baseline:           {baseline_csv}")
    print(f"Actual-writer personalization: {actual_csv}")
    print(f"Random-writer control:         {random_csv}")
    print("=" * 80)
