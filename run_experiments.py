"""Run selected HTR personalization experiments."""

from pathlib import Path
import sys

from config import (
    WRITER_IDS,
    RANDOM_SEED,
    BEGIN_SAMPLES,
    END_SAMPLES,
    STEP_SAMPLES,
    KEEP_MODELS,
    KEEP_LMDB,
    GENERIC_MODEL_PATH,
    BATCH_SIZE,
    EPOCHS,
    EVALUATION_BASELINE_RESULTS_CSV,
    ACTUAL_RESULTS_CSV,
    RANDOM_RESULTS_CSV,
    RUN_BASELINE_EVALUATION,
    RUN_REAL_PERSONALIZATION,
    RUN_RANDOM_WRITER_CONTROL,
    RUN_WORD_GROUP_PERSONALIZATION,
    WORD_GROUP_RESULTS_CSV,
    RUN_STYLE_TRANSFER_ONE_SHOT,
    RUN_STYLE_TRANSFER_FEW_SHOT,
    RUN_BIG_STYLE_TRANSFER,
    RUN_MIXED_REAL_SYNTHETIC,
    MIXED_REAL_SYNTHETIC_RATIOS,
    RUN_ORDERED_REAL_THEN_SYNTHETIC,
    ORDERED_BEGIN_SAMPLES,
    ORDERED_REAL_LIMIT,
    ORDERED_END_SAMPLES,
    ORDERED_STEP_SAMPLES,
    RUN_BINARIZATION_EXPERIMENT,
    WORD_GROUP_SELECTED_GROUPS,
)


if __name__ == "__main__":
    enabled_experiments = [
        RUN_BASELINE_EVALUATION,
        RUN_REAL_PERSONALIZATION,
        RUN_RANDOM_WRITER_CONTROL,
        RUN_WORD_GROUP_PERSONALIZATION,
        RUN_STYLE_TRANSFER_ONE_SHOT,
        RUN_STYLE_TRANSFER_FEW_SHOT,
        RUN_BIG_STYLE_TRANSFER,
        RUN_MIXED_REAL_SYNTHETIC,
        RUN_ORDERED_REAL_THEN_SYNTHETIC,
        RUN_BINARIZATION_EXPERIMENT,
    ]

    if not any(enabled_experiments):
        print("No experiment is enabled in config.py.")
        print("Enable one or more RUN_* flags, or run: python -m htr_personalization.download_resources --all")
        sys.exit(0)

    from htr_personalization.attention_htr_adapter import ensure_git_repo

    from htr_personalization.download_resources import (
        download_generic_IAM_model,
        download_real_gobo_data,
        download_synthetic_gobo_one_shot_data,
        download_synthetic_gobo_few_shot_data,
        download_big_style_transfer_data,
    )

    from htr_personalization.experiments.baseline import run_evaluation_baseline
    from htr_personalization.experiments.real_data import create_random_writer_mapping
    from htr_personalization.experiments.real_data import (
        run_real_personalization_actual_and_random,
        run_real_personalization_actual_only,
        run_random_writer_control_only,
    )
    from htr_personalization.experiments.word_groups import (
        run_word_group_personalization_experiment,
    )
    from htr_personalization.experiments.style_transfer import (
        run_selected_style_transfer_experiments,
    )
    from htr_personalization.experiments.mixed_real_synthetic import (
        run_mixed_real_synthetic_experiments,
    )
    from htr_personalization.experiments.ordered_real_then_synthetic import (
        run_ordered_real_then_synthetic_experiment,
    )
    from htr_personalization.experiments.binarization import run_binarization_experiment
    from htr_personalization.result_tables import (
        print_random_writer_mapping,
        print_baseline_summary,
        print_progressive_mean_summary,
        print_final_locations,
    )

    project_root = Path(__file__).resolve().parent
    ensure_git_repo(
        project_root,
        "AttentionHTR",
        "https://github.com/said702/AttentionHTR.git",
        "model/train.py",
    )

    print("=" * 80)
    print("Download required data and model")
    print("=" * 80)

    # The generic model is trained on IAM and is needed for all experiments as the baseline checkpoint.
    download_generic_IAM_model()

    # Real GoBo data is needed for all experiments, since evaluation is always performed on real test data.
    download_real_gobo_data(extract=True)

    if RUN_STYLE_TRANSFER_ONE_SHOT:
        download_synthetic_gobo_one_shot_data(extract=True)

    # Synthetic few-shot data is needed for normal few-shot style transfer,
    # mixed/ordered experiments, and binarized few-shot style transfer.
    if (
        RUN_STYLE_TRANSFER_FEW_SHOT
        or RUN_MIXED_REAL_SYNTHETIC
        or RUN_ORDERED_REAL_THEN_SYNTHETIC
        or RUN_BINARIZATION_EXPERIMENT
    ):
        download_synthetic_gobo_few_shot_data(extract=True)

    # Big Style Transfer / 10k data is needed for normal and binarized Big Style Transfer.
    if RUN_BIG_STYLE_TRANSFER or RUN_BINARIZATION_EXPERIMENT:
        download_big_style_transfer_data(extract=True)

    print("=" * 80)
    print("Experiment configuration")
    print("=" * 80)
    print(f"Writers: {WRITER_IDS}")
    print(f"Number of writers: {len(WRITER_IDS)}")
    print(f"begin_samples: {BEGIN_SAMPLES}")
    print(f"end_samples: {END_SAMPLES}")
    print(f"step_samples: {STEP_SAMPLES}")
    print(f"random_seed: {RANDOM_SEED}")
    print(f"keep_models: {KEEP_MODELS}")
    print(f"keep_lmdb: {KEEP_LMDB}")
    print(f"generic_model_path: {GENERIC_MODEL_PATH}")
    print(f"batch_size: {BATCH_SIZE}")
    print(f"epochs: {EPOCHS}")
    print("-" * 80)
    print(f"RUN_BASELINE_EVALUATION: {RUN_BASELINE_EVALUATION}")
    print(f"RUN_REAL_PERSONALIZATION: {RUN_REAL_PERSONALIZATION}")
    print(f"RUN_RANDOM_WRITER_CONTROL: {RUN_RANDOM_WRITER_CONTROL}")
    print(f"RUN_WORD_GROUP_PERSONALIZATION: {RUN_WORD_GROUP_PERSONALIZATION}")
    print(f"RUN_STYLE_TRANSFER_ONE_SHOT: {RUN_STYLE_TRANSFER_ONE_SHOT}")
    print(f"RUN_STYLE_TRANSFER_FEW_SHOT: {RUN_STYLE_TRANSFER_FEW_SHOT}")
    print(f"RUN_BIG_STYLE_TRANSFER: {RUN_BIG_STYLE_TRANSFER}")
    print(f"RUN_MIXED_REAL_SYNTHETIC: {RUN_MIXED_REAL_SYNTHETIC}")
    print(f"MIXED_REAL_SYNTHETIC_RATIOS: {MIXED_REAL_SYNTHETIC_RATIOS}")
    print(f"RUN_ORDERED_REAL_THEN_SYNTHETIC: {RUN_ORDERED_REAL_THEN_SYNTHETIC}")
    print(f"ORDERED_BEGIN_SAMPLES: {ORDERED_BEGIN_SAMPLES}")
    print(f"ORDERED_REAL_LIMIT: {ORDERED_REAL_LIMIT}")
    print(f"ORDERED_END_SAMPLES: {ORDERED_END_SAMPLES}")
    print(f"ORDERED_STEP_SAMPLES: {ORDERED_STEP_SAMPLES}")
    print(f"RUN_BINARIZATION_EXPERIMENT: {RUN_BINARIZATION_EXPERIMENT}")
    print("=" * 80)

    baseline_output = None
    personalization_output = None
    word_group_output = None
    style_transfer_outputs = None
    mixed_outputs = None
    ordered_output = None
    binarization_output = None

    # --------------------------------------------------
    # 1) Baseline evaluation
    # --------------------------------------------------

    if RUN_BASELINE_EVALUATION:
        print("\n" + "=" * 80)
        print("Running baseline evaluation")
        print("=" * 80)

        baseline_output = run_evaluation_baseline(
            writer_ids=WRITER_IDS,
            generic_model_path=GENERIC_MODEL_PATH,
        )

        print_baseline_summary(
            baseline_results=baseline_output["results"],
        )
    else:
        print("\nSkipping baseline evaluation.")

    # --------------------------------------------------
    # 2) Real-data personalization and/or random-writer control
    # --------------------------------------------------

    if RUN_REAL_PERSONALIZATION or RUN_RANDOM_WRITER_CONTROL:
        print("\n" + "=" * 80)
        print("Running real-data personalization/control experiments")
        print("=" * 80)

        if RUN_RANDOM_WRITER_CONTROL:
            if len(WRITER_IDS) < 2:
                raise ValueError(
                    "RUN_RANDOM_WRITER_CONTROL requires at least two writers. "
                    "Use at least two WRITER_IDS or disable the random-writer control."
                )

            random_writer_mapping = create_random_writer_mapping(
                writer_ids=WRITER_IDS,
                seed=RANDOM_SEED,
            )
            print_random_writer_mapping(random_writer_mapping)
        else:
            random_writer_mapping = {}

        if RUN_REAL_PERSONALIZATION and RUN_RANDOM_WRITER_CONTROL:
            print("Actual-writer personalization and random-writer control are both enabled.")
            print("The random-writer control will reuse the models trained for real personalization.")

            personalization_output = run_real_personalization_actual_and_random(
                writer_ids=WRITER_IDS,
                generic_model_path=GENERIC_MODEL_PATH,
                random_writer_mapping=random_writer_mapping,
                begin_samples=BEGIN_SAMPLES,
                end_samples=END_SAMPLES,
                step_samples=STEP_SAMPLES,
                seed=RANDOM_SEED,
                keep_models=KEEP_MODELS,
            )

        elif RUN_REAL_PERSONALIZATION:
            print("Running actual-writer real-data personalization only.")

            personalization_output = run_real_personalization_actual_only(
                writer_ids=WRITER_IDS,
                generic_model_path=GENERIC_MODEL_PATH,
                begin_samples=BEGIN_SAMPLES,
                end_samples=END_SAMPLES,
                step_samples=STEP_SAMPLES,
                seed=RANDOM_SEED,
                keep_models=KEEP_MODELS,
            )

        else:
            print("Running random-writer control only.")

            personalization_output = run_random_writer_control_only(
                writer_ids=WRITER_IDS,
                generic_model_path=GENERIC_MODEL_PATH,
                random_writer_mapping=random_writer_mapping,
                begin_samples=BEGIN_SAMPLES,
                end_samples=END_SAMPLES,
                step_samples=STEP_SAMPLES,
                seed=RANDOM_SEED,
                keep_models=KEEP_MODELS,
            )

        print_progressive_mean_summary(
            actual_results=personalization_output["actual_results"],
            random_results=personalization_output["random_results"],
        )
    else:
        print("\nSkipping real-data personalization and random-writer control.")

    # --------------------------------------------------
    # 3) Word-group personalization
    # --------------------------------------------------

    if RUN_WORD_GROUP_PERSONALIZATION:
        print("\n" + "=" * 80)
        print("Running word-group personalization")
        print("=" * 80)

        word_group_output = run_word_group_personalization_experiment(
            writer_ids=WRITER_IDS,
            generic_model_path=GENERIC_MODEL_PATH,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            keep_models=KEEP_MODELS,
            selected_groups=WORD_GROUP_SELECTED_GROUPS,
        )
    else:
        print("\nSkipping word-group personalization.")

    # --------------------------------------------------
    # 4) Style Transfer One-Shot / Few-Shot
    # --------------------------------------------------

    if RUN_STYLE_TRANSFER_ONE_SHOT or RUN_STYLE_TRANSFER_FEW_SHOT:
        print("\n" + "=" * 80)
        print("Running style-transfer one-shot/few-shot personalization")
        print("=" * 80)

        style_transfer_outputs = run_selected_style_transfer_experiments(
            writer_ids=WRITER_IDS,
            generic_model_path=GENERIC_MODEL_PATH,
            run_one_shot=RUN_STYLE_TRANSFER_ONE_SHOT,
            run_few_shot=RUN_STYLE_TRANSFER_FEW_SHOT,
            run_big_style_transfer=False,
            keep_models=KEEP_MODELS,
        )
    else:
        print("\nSkipping style-transfer one-shot/few-shot personalization.")

    # --------------------------------------------------
    # 5) Mixed real/synthetic personalization
    # --------------------------------------------------

    if RUN_MIXED_REAL_SYNTHETIC:
        print("\n" + "=" * 80)
        print("Running mixed real/synthetic personalization")
        print("=" * 80)

        mixed_outputs = run_mixed_real_synthetic_experiments(
            writer_ids=WRITER_IDS,
            generic_model_path=GENERIC_MODEL_PATH,
            mixed_ratios=MIXED_REAL_SYNTHETIC_RATIOS,
            seed=RANDOM_SEED,
            keep_models=KEEP_MODELS,
        )
    else:
        print("\nSkipping mixed real/synthetic personalization.")

    # --------------------------------------------------
    # 6) Ordered real-then-synthetic personalization
    # --------------------------------------------------

    if RUN_ORDERED_REAL_THEN_SYNTHETIC:
        print("\n" + "=" * 80)
        print("Running ordered real-then-synthetic personalization")
        print("=" * 80)

        ordered_output = run_ordered_real_then_synthetic_experiment(
            writer_ids=WRITER_IDS,
            generic_model_path=GENERIC_MODEL_PATH,
            begin_samples=ORDERED_BEGIN_SAMPLES,
            real_limit=ORDERED_REAL_LIMIT,
            end_samples=ORDERED_END_SAMPLES,
            step_samples=ORDERED_STEP_SAMPLES,
            seed=RANDOM_SEED,
            keep_models=KEEP_MODELS,
        )
    else:
        print("\nSkipping ordered real-then-synthetic personalization.")

    # --------------------------------------------------
    # 7) Big Style Transfer
    # --------------------------------------------------

    if RUN_BIG_STYLE_TRANSFER:
        print("\n" + "=" * 80)
        print("Running Big Style Transfer personalization")
        print("=" * 80)

        big_style_transfer_output = run_selected_style_transfer_experiments(
            writer_ids=WRITER_IDS,
            generic_model_path=GENERIC_MODEL_PATH,
            run_one_shot=False,
            run_few_shot=False,
            run_big_style_transfer=True,
            keep_models=KEEP_MODELS,
        )

        if style_transfer_outputs is None:
            style_transfer_outputs = {}

        style_transfer_outputs.update(big_style_transfer_output)
    else:
        print("\nSkipping Big Style Transfer personalization.")

    # --------------------------------------------------
    # 8) Binarization experiment
    # --------------------------------------------------

    if RUN_BINARIZATION_EXPERIMENT:
        print("\n" + "=" * 80)
        print("Running binarization experiment")
        print("=" * 80)

        binarization_output = run_binarization_experiment(
            writer_ids=WRITER_IDS,
            generic_model_path=GENERIC_MODEL_PATH,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            keep_models=KEEP_MODELS,
        )
    else:
        print("\nSkipping binarization experiment.")

    # --------------------------------------------------
    # 9) Final paths
    # --------------------------------------------------

    print("\n" + "=" * 80)
    print("Final result locations")
    print("=" * 80)

    print_final_locations(
        baseline_csv=EVALUATION_BASELINE_RESULTS_CSV,
        actual_csv=ACTUAL_RESULTS_CSV,
        random_csv=RANDOM_RESULTS_CSV,
    )

    if word_group_output:
        print("\nWord-group personalization CSV file:")
        print(f"{word_group_output['experiment']}: {word_group_output['csv']}")

    if style_transfer_outputs:
        print("\nStyle-transfer CSV files:")
        for experiment_name, output in style_transfer_outputs.items():
            print(f"{experiment_name}: {output['csv']}")

    if mixed_outputs:
        print("\nMixed real/synthetic CSV files:")
        for ratio_name, output in mixed_outputs.items():
            print(f"{ratio_name}: {output['csv']}")

    if ordered_output:
        print("\nOrdered real-then-synthetic CSV file:")
        print(f"{ordered_output['experiment']}: {ordered_output['csv']}")

    if binarization_output:
        print("\nBinarization experiment CSV files:")
        for experiment_name, output in binarization_output["outputs"].items():
            print(f"{experiment_name}: {output['csv']}")

    print("\nFinished.")
