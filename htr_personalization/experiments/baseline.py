"""Baseline evaluation of the generic AttentionHTR model."""

from config import (
    TEST_SAMPLES,
    EVALUATION_BASELINE_RESULTS_CSV,
)
from htr_personalization.attention_htr_adapter import evaluate_model_on_writer
from htr_personalization.result_tables import append_result_and_save


EXPERIMENT_NAME = "Evaluation_Baseline"


# Generic model baseline evaluation

def run_evaluation_baseline(writer_ids, generic_model_path):
    """
    Evaluates the generic model on every writer without fine-tuning.

    This is the baseline before personalization:
    generic model -> test writer X.
    """
    print("=" * 80)
    print(EXPERIMENT_NAME)
    print("=" * 80)
    print("Generic model is evaluated on all writers without fine-tuning.")
    print("=" * 80)

    results = []

    for writer_id in writer_ids:
        print("\n" + "#" * 80)
        print(f"Writer {writer_id}: generic model evaluation")
        print("#" * 80)

        cer_value, _, test_path = evaluate_model_on_writer(
            writer_id=writer_id,
            model_path=generic_model_path,
            test_samples=TEST_SAMPLES,
        )

        row = {
            "experiment": EXPERIMENT_NAME,
            "writer_id": writer_id,
            "training_writer_id": "generic",
            "test_samples": TEST_SAMPLES,
            "cer": cer_value,
            "model_path": str(generic_model_path),
            "test_path": str(test_path),
        }

        append_result_and_save(
            path=EVALUATION_BASELINE_RESULTS_CSV,
            rows=results,
            row=row,
        )

        print(
            f"Result | writer_id={writer_id} | "
            f"training_writer_id=generic | "
            f"cer={cer_value:.4f}"
        )

    return {
        "results": results,
    }
