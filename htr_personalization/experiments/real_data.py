"""Real-data personalization and random-writer control experiments."""

import random
from collections import defaultdict

from config import (
    PROJECT_ROOT,
    RANDOM_SEED,
    BEGIN_SAMPLES,
    END_SAMPLES,
    STEP_SAMPLES,
    BATCH_SIZE,
    EPOCHS,
    KEEP_MODELS,
    TEST_SAMPLES,
    ACTUAL_RESULTS_CSV,
    RANDOM_RESULTS_CSV,
)
from htr_personalization.attention_htr_adapter import (
    process_training_set,
    get_best_accuracy_model,
    delete_model_checkpoints,
    evaluate_model_on_writer,
)
from htr_personalization.gobo_preparation import (
    get_writer_progressive_data,
    prepare_progressive_train_folder,
)
from htr_personalization.result_tables import append_result_and_save


RANDOM_WRITER_EXPERIMENT_NAME = "RealData_Random_Writer"


# Random writer mapping

def create_random_writer_mapping(writer_ids, seed=RANDOM_SEED):
    """
    Creates a derangement:
    no writer is mapped to itself.

    Mapping direction:
    target_writer_id -> random_training_writer_id
    """
    writer_ids = list(writer_ids)
    source_writer_ids = writer_ids.copy()
    rng = random.Random(seed)

    while True:
        rng.shuffle(source_writer_ids)

        if all(target != source for target, source in zip(writer_ids, source_writer_ids)):
            return dict(zip(writer_ids, source_writer_ids))


def get_nearest_available_sample_size(source_sample_sizes, requested_sample_size):
    """
    Returns the largest source sample size <= requested_sample_size.

    This is needed when the target writer has, for example, 528 samples
    but the random source writer has only 522 samples.
    """
    valid_sizes = [size for size in source_sample_sizes if size <= requested_sample_size]

    if len(valid_sizes) == 0:
        raise ValueError(
            f"No source sample size available for requested_sample_size={requested_sample_size}"
        )

    return max(valid_sizes)


def build_random_evaluation_plan(sample_sizes_by_writer, random_writer_mapping):
    """
    Builds a source-centered plan.

    This allows KEEP_MODELS=False:
    once a source writer model is trained, all random-target evaluations using
    that model are executed immediately, then the model can be deleted.
    """
    plan = defaultdict(list)

    for target_writer_id, source_writer_id in random_writer_mapping.items():
        target_sample_sizes = sample_sizes_by_writer[target_writer_id]
        source_sample_sizes = sample_sizes_by_writer[source_writer_id]

        for target_sample_size in target_sample_sizes:
            source_sample_size = get_nearest_available_sample_size(
                source_sample_sizes=source_sample_sizes,
                requested_sample_size=target_sample_size,
            )

            plan[(source_writer_id, source_sample_size)].append(
                {
                    "target_writer_id": target_writer_id,
                    "source_writer_id": source_writer_id,
                    "target_sample_size": target_sample_size,
                    "source_sample_size": source_sample_size,
                }
            )

    return dict(plan)


# Random-writer evaluation for one trained model

def evaluate_random_writer_model_for_plan(
    source_writer_id,
    source_sample_size,
    model_path,
    random_evaluation_plan,
    rows,
):
    plan_key = (source_writer_id, source_sample_size)
    plan_items = random_evaluation_plan.get(plan_key, [])

    evaluated_rows = []

    for item in plan_items:
        target_writer_id = item["target_writer_id"]
        target_sample_size = item["target_sample_size"]

        cer_value, _, test_path = evaluate_model_on_writer(
            writer_id=target_writer_id,
            model_path=model_path,
            test_samples=TEST_SAMPLES,
        )

        row = {
            "experiment": RANDOM_WRITER_EXPERIMENT_NAME,
            "target_writer_id": target_writer_id,
            "training_writer_id": source_writer_id,
            "target_sample_size": target_sample_size,
            "training_sample_size": source_sample_size,
            "test_samples": TEST_SAMPLES,
            "cer": cer_value,
            "model_path": str(model_path),
            "test_path": str(test_path),
        }

        append_result_and_save(
            path=RANDOM_RESULTS_CSV,
            rows=rows,
            row=row,
        )

        evaluated_rows.append(row)

        print(
            f"Random result | target_writer_id={target_writer_id} | "
            f"training_writer_id={source_writer_id} | "
            f"target_sample_size={target_sample_size} | "
            f"training_sample_size={source_sample_size} | "
            f"cer={cer_value:.4f}"
        )

    return evaluated_rows


ACTUAL_WRITER_EXPERIMENT_NAME = "RealData_Actual_Writer"


# Actual-writer evaluation for one trained model

def evaluate_actual_writer_model(
    writer_id,
    sample_size,
    available_train_samples,
    model_path,
    train_path,
    rows,
):
    cer_value, _, test_path = evaluate_model_on_writer(
        writer_id=writer_id,
        model_path=model_path,
        test_samples=TEST_SAMPLES,
    )

    row = {
        "experiment": ACTUAL_WRITER_EXPERIMENT_NAME,
        "writer_id": writer_id,
        "training_writer_id": writer_id,
        "sample_size": sample_size,
        "available_train_samples": available_train_samples,
        "test_samples": TEST_SAMPLES,
        "cer": cer_value,
        "model_path": str(model_path),
        "train_path": str(train_path),
        "test_path": str(test_path),
    }

    append_result_and_save(
        path=ACTUAL_RESULTS_CSV,
        rows=rows,
        row=row,
    )

    print(
        f"Actual result | writer_id={writer_id} | "
        f"sample_size={sample_size} | "
        f"cer={cer_value:.4f}"
    )

    return row


# Storage-efficient progressive personalization runner

def collect_progressive_metadata(
    writer_ids,
    seed=RANDOM_SEED,
    begin_samples=BEGIN_SAMPLES,
    end_samples=END_SAMPLES,
    step_samples=STEP_SAMPLES,
):
    progressive_data_by_writer = {}
    sample_sizes_by_writer = {}
    available_samples_by_writer = {}

    for writer_id in writer_ids:
        writer_data = get_writer_progressive_data(
            project_root=PROJECT_ROOT,
            writer_id=writer_id,
            seed=seed,
            begin_samples=begin_samples,
            end_samples=end_samples,
            step_samples=step_samples,
        )

        progressive_data_by_writer[writer_id] = writer_data
        sample_sizes_by_writer[writer_id] = writer_data["sample_sizes"]
        available_samples_by_writer[writer_id] = writer_data["available_samples"]

    return {
        "progressive_data_by_writer": progressive_data_by_writer,
        "sample_sizes_by_writer": sample_sizes_by_writer,
        "available_samples_by_writer": available_samples_by_writer,
    }


def run_real_personalization_actual_and_random(
    writer_ids,
    generic_model_path,
    random_writer_mapping,
    begin_samples=BEGIN_SAMPLES,
    end_samples=END_SAMPLES,
    step_samples=STEP_SAMPLES,
    seed=RANDOM_SEED,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    keep_models=KEEP_MODELS,
):
    """
    Runs both personalization-real experiments in a storage-efficient way.

    For each trained model:
    1. evaluate actual writer
    2. evaluate all random-writer targets that need this model
    3. delete checkpoint files if keep_models=False

    This avoids storing thousands of 200 MB checkpoints at once.
    """
    print("=" * 80)
    print("Personalization real experiments")
    print("=" * 80)
    print("Running Actual Writer and Random Writer with one shared training loop.")
    print(f"begin_samples: {begin_samples}")
    print(f"end_samples: {end_samples}")
    print(f"step_samples: {step_samples}")
    print(f"seed: {seed}")
    print(f"keep_models: {keep_models}")
    print("=" * 80)

    metadata = collect_progressive_metadata(
        writer_ids=writer_ids,
        seed=seed,
        begin_samples=begin_samples,
        end_samples=end_samples,
        step_samples=step_samples,
    )

    progressive_data_by_writer = metadata["progressive_data_by_writer"]
    sample_sizes_by_writer = metadata["sample_sizes_by_writer"]
    available_samples_by_writer = metadata["available_samples_by_writer"]

    random_evaluation_plan = build_random_evaluation_plan(
        sample_sizes_by_writer=sample_sizes_by_writer,
        random_writer_mapping=random_writer_mapping,
    )

    actual_results = []
    random_results = []

    for training_writer_id in writer_ids:
        writer_data = progressive_data_by_writer[training_writer_id]
        ordered_entries = writer_data["ordered_entries"]
        sample_sizes = writer_data["sample_sizes"]
        available_samples = writer_data["available_samples"]

        print("\n" + "#" * 80)
        print(f"Training writer {training_writer_id}")
        print(f"Available train samples: {available_samples}")
        print(f"Progressive sample sizes: {sample_sizes}")
        print("#" * 80)

        for sample_size in sample_sizes:
            print("\n" + "-" * 80)
            print(f"Training writer {training_writer_id} | samples_{sample_size}")
            print("-" * 80)

            train_path = prepare_progressive_train_folder(
                project_root=PROJECT_ROOT,
                writer_id=training_writer_id,
                ordered_entries=ordered_entries,
                sample_size=sample_size,
            )

            trained_model_folder = process_training_set(
                input_path=train_path,
                pretrained_model_path=generic_model_path,
                batch_size=batch_size,
                epochs=epochs,
            )

            saved_model_path = get_best_accuracy_model(trained_model_folder)

            evaluate_actual_writer_model(
                writer_id=training_writer_id,
                sample_size=sample_size,
                available_train_samples=available_samples,
                model_path=saved_model_path,
                train_path=train_path,
                rows=actual_results,
            )

            evaluate_random_writer_model_for_plan(
                source_writer_id=training_writer_id,
                source_sample_size=sample_size,
                model_path=saved_model_path,
                random_evaluation_plan=random_evaluation_plan,
                rows=random_results,
            )

            delete_model_checkpoints(
                model_path=saved_model_path,
                keep_models=keep_models,
            )

    return {
        "actual_results": actual_results,
        "random_results": random_results,
        "sample_sizes_by_writer": sample_sizes_by_writer,
        "available_samples_by_writer": available_samples_by_writer,
        "random_writer_mapping": random_writer_mapping,
    }



def run_random_writer_control_only(
    writer_ids,
    generic_model_path,
    random_writer_mapping,
    begin_samples=BEGIN_SAMPLES,
    end_samples=END_SAMPLES,
    step_samples=STEP_SAMPLES,
    seed=RANDOM_SEED,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    keep_models=KEEP_MODELS,
):
    """
    Runs only the random-writer control experiment from the paper.

    This trains models on samples from the mapped source writers and evaluates
    them on different target writers. If RUN_REAL_PERSONALIZATION and
    RUN_RANDOM_WRITER_CONTROL are both enabled, use
    run_real_personalization_actual_and_random instead, because it evaluates the
    random-writer control directly with the models trained for real
    personalization.
    """

    if len(writer_ids) < 2:
        raise ValueError(
            "Random-writer control requires at least two writers. "
            "Use at least two WRITER_IDS or disable RUN_RANDOM_WRITER_CONTROL."
        )

    print("=" * 80)
    print("Random-writer control")
    print("=" * 80)
    print("Running failed 'personalization' with samples from a different writer.")
    print(f"begin_samples: {begin_samples}")
    print(f"end_samples: {end_samples}")
    print(f"step_samples: {step_samples}")
    print(f"seed: {seed}")
    print(f"keep_models: {keep_models}")
    print("=" * 80)

    metadata = collect_progressive_metadata(
        writer_ids=writer_ids,
        seed=seed,
        begin_samples=begin_samples,
        end_samples=end_samples,
        step_samples=step_samples,
    )

    progressive_data_by_writer = metadata["progressive_data_by_writer"]
    sample_sizes_by_writer = metadata["sample_sizes_by_writer"]
    available_samples_by_writer = metadata["available_samples_by_writer"]

    random_evaluation_plan = build_random_evaluation_plan(
        sample_sizes_by_writer=sample_sizes_by_writer,
        random_writer_mapping=random_writer_mapping,
    )

    random_results = []

    for training_writer_id in writer_ids:
        writer_data = progressive_data_by_writer[training_writer_id]
        ordered_entries = writer_data["ordered_entries"]
        sample_sizes = writer_data["sample_sizes"]
        available_samples = writer_data["available_samples"]

        print("\n" + "#" * 80)
        print(f"Training source writer {training_writer_id} for random-writer control")
        print(f"Available train samples: {available_samples}")
        print(f"Progressive sample sizes: {sample_sizes}")
        print("#" * 80)

        for sample_size in sample_sizes:
            plan_key = (training_writer_id, sample_size)

            if plan_key not in random_evaluation_plan:
                print(
                    f"Skipping writer {training_writer_id} | samples_{sample_size}: "
                    "not needed by the random-writer control plan."
                )
                continue

            print("\n" + "-" * 80)
            print(
                f"Training source writer {training_writer_id} | "
                f"samples_{sample_size} for random-writer control"
            )
            print("-" * 80)

            train_path = prepare_progressive_train_folder(
                project_root=PROJECT_ROOT,
                writer_id=training_writer_id,
                ordered_entries=ordered_entries,
                sample_size=sample_size,
            )

            trained_model_folder = process_training_set(
                input_path=train_path,
                pretrained_model_path=generic_model_path,
                batch_size=batch_size,
                epochs=epochs,
            )

            saved_model_path = get_best_accuracy_model(trained_model_folder)

            evaluate_random_writer_model_for_plan(
                source_writer_id=training_writer_id,
                source_sample_size=sample_size,
                model_path=saved_model_path,
                random_evaluation_plan=random_evaluation_plan,
                rows=random_results,
            )

            delete_model_checkpoints(
                model_path=saved_model_path,
                keep_models=keep_models,
            )

    return {
        "actual_results": [],
        "random_results": random_results,
        "sample_sizes_by_writer": sample_sizes_by_writer,
        "available_samples_by_writer": available_samples_by_writer,
        "random_writer_mapping": random_writer_mapping,
    }

def run_real_personalization_actual_only(
    writer_ids,
    generic_model_path,
    begin_samples=BEGIN_SAMPLES,
    end_samples=END_SAMPLES,
    step_samples=STEP_SAMPLES,
    seed=RANDOM_SEED,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    keep_models=KEEP_MODELS,
):
    """
    Runs only the actual-writer real personalization experiment.
    This is useful when only one writer is selected, because random-writer
    evaluation needs at least two writers.
    """

    print("=" * 80)
    print("Personalization real experiment")
    print("=" * 80)
    print("Running Actual Writer only.")
    print(f"begin_samples: {begin_samples}")
    print(f"end_samples: {end_samples}")
    print(f"step_samples: {step_samples}")
    print(f"seed: {seed}")
    print(f"keep_models: {keep_models}")
    print("=" * 80)

    metadata = collect_progressive_metadata(
        writer_ids=writer_ids,
        seed=seed,
        begin_samples=begin_samples,
        end_samples=end_samples,
        step_samples=step_samples,
    )

    progressive_data_by_writer = metadata["progressive_data_by_writer"]
    sample_sizes_by_writer = metadata["sample_sizes_by_writer"]
    available_samples_by_writer = metadata["available_samples_by_writer"]

    actual_results = []

    for training_writer_id in writer_ids:
        writer_data = progressive_data_by_writer[training_writer_id]
        ordered_entries = writer_data["ordered_entries"]
        sample_sizes = writer_data["sample_sizes"]
        available_samples = writer_data["available_samples"]

        print("\n" + "#" * 80)
        print(f"Training writer {training_writer_id}")
        print(f"Available train samples: {available_samples}")
        print(f"Progressive sample sizes: {sample_sizes}")
        print("#" * 80)

        for sample_size in sample_sizes:
            print("\n" + "-" * 80)
            print(f"Training writer {training_writer_id} | samples_{sample_size}")
            print("-" * 80)

            train_path = prepare_progressive_train_folder(
                project_root=PROJECT_ROOT,
                writer_id=training_writer_id,
                ordered_entries=ordered_entries,
                sample_size=sample_size,
            )

            trained_model_folder = process_training_set(
                input_path=train_path,
                pretrained_model_path=generic_model_path,
                batch_size=batch_size,
                epochs=epochs,
            )

            saved_model_path = get_best_accuracy_model(trained_model_folder)

            evaluate_actual_writer_model(
                writer_id=training_writer_id,
                sample_size=sample_size,
                available_train_samples=available_samples,
                model_path=saved_model_path,
                train_path=train_path,
                rows=actual_results,
            )

            delete_model_checkpoints(
                model_path=saved_model_path,
                keep_models=keep_models,
            )

    return {
        "actual_results": actual_results,
        "random_results": [],
        "sample_sizes_by_writer": sample_sizes_by_writer,
        "available_samples_by_writer": available_samples_by_writer,
        "random_writer_mapping": {},
    }