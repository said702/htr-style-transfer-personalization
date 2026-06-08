"""Shared helpers for data preparation, training, and evaluation."""

import os
import sys
import shutil
import random
import string
import subprocess
import importlib.util
from pathlib import Path

from config import TEST_SAMPLES, SHOW_PREDICTION_PROGRESS, KEEP_LMDB


def delete_lmdb_cache(*lmdb_paths, keep_lmdb=False):
    """Delete temporary LMDB folders created for AttentionHTR training.

    LMDB folders are training caches. They are not needed after the
    corresponding model has been trained, so the default behavior is to remove
    them immediately.
    """
    if keep_lmdb:
        return []

    deleted_paths = []

    for lmdb_path in lmdb_paths:
        lmdb_path = Path(lmdb_path)

        if lmdb_path.exists():
            shutil.rmtree(lmdb_path)
            deleted_paths.append(lmdb_path)

    if deleted_paths:
        print("Deleted LMDB cache folders:")
        for deleted_path in deleted_paths:
            print(f"  {deleted_path}")

    return deleted_paths


def recreate_lmdb_folder(lmdb_path):
    """Create an empty LMDB output folder, replacing an older cache if present."""
    lmdb_path = Path(lmdb_path)

    if lmdb_path.exists():
        shutil.rmtree(lmdb_path)

    lmdb_path.mkdir(parents=True, exist_ok=True)
    return lmdb_path


def process_training_set(input_path, pretrained_model_path, batch_size=10, epochs=5):
    input_path = Path(input_path)
    pretrained_model = Path(pretrained_model_path)

    writer_folder = input_path.parent.name
    sample_folder = input_path.name

    project_root = Path(__file__).resolve().parents[1]

    input_path_str = str(input_path)

    is_real_data = (
        "train_real_data" in input_path_str
        or "progressive_real_train" in input_path_str
    )

    if is_real_data:
        real_experiment_root = "Real_Data"
        exp_name = f"{real_experiment_root}/{writer_folder}/{sample_folder}"
    else:
        exp_name = f"Synthetic_Data/{writer_folder}/{sample_folder}"

    attention_model_dir = project_root / "AttentionHTR" / "model"
    create_lmdb_script = attention_model_dir / "create_lmdb_dataset.py"
    train_script = attention_model_dir / "train.py"

    output_root = project_root / "lmdb_data"

    if is_real_data:
        out_train = output_root / real_experiment_root / writer_folder / sample_folder / "train_lmdb"
        out_val = output_root / real_experiment_root / writer_folder / sample_folder / "val_lmdb"

        expected_model_folder = (
            project_root
            / "saved_models"
            / real_experiment_root
            / writer_folder
            / sample_folder
        )
    else:
        out_train = output_root / "Synthetic_Data" / writer_folder / sample_folder / "train_lmdb"
        out_val = output_root / "Synthetic_Data" / writer_folder / sample_folder / "val_lmdb"

        expected_model_folder = (
            project_root
            / "saved_models"
            / "Synthetic_Data"
            / writer_folder
            / sample_folder
        )

    gt_train = input_path / "gt.txt"
    gt_val = gt_train

    if not input_path.exists():
        raise FileNotFoundError(f"Input folder not found: {input_path}")

    if not gt_train.exists():
        raise FileNotFoundError(f"gt.txt not found: {gt_train}")

    if not attention_model_dir.exists():
        raise FileNotFoundError(f"AttentionHTR folder not found: {attention_model_dir}")

    if not create_lmdb_script.exists():
        raise FileNotFoundError(f"create_lmdb_dataset.py not found: {create_lmdb_script}")

    if not train_script.exists():
        raise FileNotFoundError(f"train.py not found: {train_script}")

    if not pretrained_model.exists():
        raise FileNotFoundError(f"Pretrained model not found: {pretrained_model}")

    image_extensions = (".jpg", ".jpeg", ".png")

    image_count = len([
        f for f in os.listdir(input_path)
        if f.lower().endswith(image_extensions)
    ])

    with open(gt_train, "r", encoding="utf-8") as f:
        gt_lines = sum(1 for line in f if line.strip())

    if image_count == 0:
        raise ValueError(f"No images found in: {input_path}")

    iters_per_epoch = max(1, image_count // batch_size)
    num_iter = epochs * iters_per_epoch

    print("=" * 80)
    print(f"Input folder: {input_path}")
    print(f"Writer: {writer_folder}")
    print(f"Samples: {sample_folder}")
    print(f"Experiment: {exp_name}")
    print(f"Images: {image_count}")
    print(f"Pretrained model: {pretrained_model}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"num_iter: {num_iter}")
    print(f"Train LMDB: {out_train}")
    print(f"Val LMDB: {out_val}")
    print(f"Expected model folder: {expected_model_folder}")

    if image_count != gt_lines:
        print("Warning: image count and GT lines do not match")

    print("=" * 80)

    recreate_lmdb_folder(out_train)
    recreate_lmdb_folder(out_val)

    def run_command(command, cwd):
        print("\n" + "-" * 80)
        print(" ".join(str(part) for part in command))
        print("-" * 80)

        subprocess.run(
            command,
            cwd=cwd,
            check=True
        )

    create_lmdb_train = [
        sys.executable,
        str(create_lmdb_script),
        "--inputPath", str(input_path),
        "--gtFile", str(gt_train),
        "--outputPath", str(out_train)
    ]

    create_lmdb_val = [
        sys.executable,
        str(create_lmdb_script),
        "--inputPath", str(input_path),
        "--gtFile", str(gt_val),
        "--outputPath", str(out_val)
    ]

    train_command = [
        sys.executable,
        str(train_script),
        "--exp_name", exp_name,
        "--train_data", str(out_train),
        "--valid_data", str(out_val),
        "--select_data", "/",
        "--batch_ratio", "1",
        "--manualSeed", "1",
        "--Transformation", "TPS",
        "--FeatureExtraction", "ResNet",
        "--SequenceModeling", "BiLSTM",
        "--Prediction", "Attn",
        "--sensitive",
        "--num_iter", str(num_iter),
        "--batch_size", str(batch_size),
        "--saved_model", str(pretrained_model)
    ]

    try:
        print("Creating train LMDB...")
        run_command(create_lmdb_train, cwd=attention_model_dir)

        print("Creating validation LMDB...")
        run_command(create_lmdb_val, cwd=attention_model_dir)

        print("Starting training...")
        run_command(train_command, cwd=project_root)

    finally:
        delete_lmdb_cache(out_train, out_val, keep_lmdb=KEEP_LMDB)

    print("=" * 80)
    print(f"Done: {exp_name}")
    print(f"Images: {image_count}")
    print(f"Train LMDB: {out_train}")
    print(f"Val LMDB: {out_val}")
    print(f"Model folder: {expected_model_folder}")
    print("=" * 80)

    return expected_model_folder


def build_real_gobo_data(writer_id, num_samples, split="train", seed=1):
    project_root = Path(__file__).resolve().parents[1]

    gobo_words_root = (
        project_root
        / "data"
        / "real"
        / "GoBo_v1-0"
        / "words"
        / str(writer_id)
    )

    split = split.lower().strip()

    if split == "train":
        dataset_names = [
            "brown",
            "cedar",
            "nonwords",
            "domain_A_train",
            "domain_B_train"
        ]
        output_base_folder = "train_real_data"

    elif split == "test":
        dataset_names = [
            "domain_A_test",
            "domain_B_test"
        ]
        output_base_folder = "test_real_data"

    else:
        raise ValueError("split must be either 'train' or 'test'")

    output_root = (
        project_root
        / output_base_folder
        / f"writer_{writer_id}"
        / f"samples_{num_samples}"
    )

    gt_output_path = output_root / "gt.txt"
    image_extensions = (".jpg", ".jpeg", ".png")

    if not gobo_words_root.exists():
        raise FileNotFoundError(
            f"GoBo writer folder not found:\n{gobo_words_root}\n\n"
            f"Project root:\n{project_root}"
        )

    if output_root.exists():
        shutil.rmtree(output_root)

    output_root.mkdir(parents=True, exist_ok=True)

    all_entries = []

    for dataset_name in dataset_names:
        txt_path = gobo_words_root / f"{dataset_name}.txt"

        if not txt_path.exists():
            print(f"Warning: TXT file missing, skipped: {txt_path}")
            continue

        print(f"Reading: {txt_path}")

        with open(txt_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()

                if not line:
                    continue

                parts = line.split(maxsplit=2)

                if len(parts) != 3:
                    print(f"Warning: invalid line {line_num} in {txt_path}: {line}")
                    continue

                relative_img_path, status, word = parts

                if status not in {"ok", "rw"}:
                    continue

                if not relative_img_path.lower().endswith(image_extensions):
                    continue

                src_img_path = gobo_words_root / relative_img_path

                if not src_img_path.exists():
                    print(f"Warning: image missing: {src_img_path}")
                    continue

                original_filename = Path(relative_img_path).name

                all_entries.append({
                    "src_img_path": src_img_path,
                    "original_filename": original_filename,
                    "word": word,
                    "dataset_name": dataset_name
                })

    available_samples = len(all_entries)

    if available_samples == 0:
        raise ValueError("No valid 'ok' samples found.")

    requested_samples = num_samples

    if num_samples > available_samples:
        print(
            f"Warning: requested {num_samples} samples, "
            f"but only {available_samples} are available. Using all samples."
        )
        num_samples = available_samples

    print("=" * 80)
    print(f"Writer: {writer_id}")
    print(f"Split: {split}")
    print(f"Available samples: {available_samples}")
    print(f"Requested samples: {requested_samples}")
    print(f"Used samples: {num_samples}")
    print(f"Output base folder: {output_base_folder}")
    print("=" * 80)

    random.seed(seed)
    selected_entries = random.sample(all_entries, num_samples)

    used_filenames = set()
    gt_lines = []
    copied_count = 0

    for entry in selected_entries:
        src_img_path = entry["src_img_path"]
        original_filename = entry["original_filename"]
        word = entry["word"]
        dataset_name = entry["dataset_name"]

        output_filename = original_filename

        if output_filename in used_filenames:
            stem = Path(original_filename).stem
            ext = Path(original_filename).suffix
            output_filename = f"{dataset_name}_{stem}{ext}"

            counter = 1
            while output_filename in used_filenames:
                output_filename = f"{dataset_name}_{stem}_{counter}{ext}"
                counter += 1

        used_filenames.add(output_filename)

        dst_img_path = output_root / output_filename
        shutil.copy2(src_img_path, dst_img_path)

        gt_lines.append(f"{output_filename}\t{word}")
        copied_count += 1

    with open(gt_output_path, "w", encoding="utf-8") as f:
        for gt_line in gt_lines:
            f.write(gt_line + "\n")

    image_count = len([
        f for f in os.listdir(output_root)
        if f.lower().endswith(image_extensions)
    ])

    with open(gt_output_path, "r", encoding="utf-8") as f:
        gt_count = sum(1 for line in f if line.strip())

    print("\n" + "=" * 80)
    print("Dataset created successfully")
    print("=" * 80)
    print(f"Writer: {writer_id}")
    print(f"Split: {split}")
    print(f"Output folder: {output_root}")
    print(f"GT file: {gt_output_path}")
    print(f"Copied images: {copied_count}")
    print(f"Images in output folder: {image_count}")

    if image_count != gt_count:
        print("Warning: image count and GT lines do not match")

    print("=" * 80)

    return output_root


def predict_attentionhtr_folder(saved_model_path, image_folder, show_progress=True):
    import torch
    from PIL import Image
    from torchvision import transforms
    from tqdm import tqdm

    project_root = Path(__file__).resolve().parents[1]
    attention_model_dir = project_root / "AttentionHTR" / "model"

    sys.path.insert(0, str(attention_model_dir))

    from model import Model

    utils_path = attention_model_dir / "utils.py"
    spec = importlib.util.spec_from_file_location("attentionhtr_utils", utils_path)
    attentionhtr_utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(attentionhtr_utils)

    AttnLabelConverter = attentionhtr_utils.AttnLabelConverter

    saved_model_path = Path(saved_model_path)
    image_folder = Path(image_folder)

    if not saved_model_path.exists():
        raise FileNotFoundError(f"Model not found: {saved_model_path}")

    if not image_folder.exists():
        raise FileNotFoundError(f"Image folder not found: {image_folder}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    character = string.printable[:-6]
    converter = AttnLabelConverter(character)

    class Config:
        pass

    opt = Config()
    opt.imgH = 32
    opt.imgW = 100
    opt.num_fiducial = 20
    opt.input_channel = 1
    opt.output_channel = 512
    opt.hidden_size = 256
    opt.batch_max_length = 25
    opt.Transformation = "TPS"
    opt.FeatureExtraction = "ResNet"
    opt.SequenceModeling = "BiLSTM"
    opt.Prediction = "Attn"
    opt.num_class = len(converter.character)

    model = Model(opt)
    model = torch.nn.DataParallel(model).to(device)

    model.load_state_dict(torch.load(saved_model_path, map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((opt.imgH, opt.imgW)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    image_extensions = (".jpg", ".jpeg", ".png")
    image_names = sorted([
        f for f in os.listdir(image_folder)
        if f.lower().endswith(image_extensions)
    ])

    results = []

    progress_bar = tqdm(
        image_names,
        desc="Predicting",
        unit="image",
        dynamic_ncols=True,
        disable=not show_progress
    )

    for image_name in progress_bar:
        image_path = image_folder / image_name

        image = Image.open(image_path).convert("L")
        image = transform(image).unsqueeze(0).to(device)

        batch_size = image.size(0)
        length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
        text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

        with torch.no_grad():
            preds = model(image, text_for_pred, is_train=False)

        _, preds_index = preds.max(2)
        preds_str = converter.decode(preds_index, length_for_pred)

        prediction = preds_str[0]
        eos_index = prediction.find("[s]")

        if eos_index != -1:
            prediction = prediction[:eos_index]

        prediction = prediction.strip()

        results.append({
            "image": image_name,
            "prediction": prediction
        })

    return results


def calculate_cer_from_results(results, gt_path):
    from jiwer import cer

    gt_path = Path(gt_path)

    if not gt_path.exists():
        raise FileNotFoundError(f"GT file not found: {gt_path}")

    gt_dict = {}

    with open(gt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            parts = line.split("\t", 1)

            if len(parts) != 2:
                continue

            image_name, gt_text = parts
            gt_dict[image_name.strip()] = gt_text.strip()

    cer_values = []
    detailed_results = []

    for item in results:
        image_name = item["image"]
        prediction = item["prediction"]

        if image_name not in gt_dict:
            continue

        gt_text = gt_dict[image_name]
        cer_value = cer(gt_text, prediction)

        cer_values.append(cer_value)

        detailed_results.append({
            "image": image_name,
            "ground_truth": gt_text,
            "prediction": prediction,
            "cer": cer_value
        })

    if len(cer_values) == 0:
        raise ValueError("No matching images found between results and GT file.")

    mean_cer = float(sum(cer_values) / len(cer_values))

    return mean_cer, detailed_results



def ensure_git_repo(project_root, repo_name, repo_url, marker_file=None):
    """
    Clone a GitHub repo into project_root/repo_name if it does not exist.

    Parameters
    ----------
    project_root : Path
        Root folder of your project.
    repo_name : str
        Local folder name, e.g. "AttentionHTR".
    repo_url : str
        GitHub URL.
    marker_file : str, optional
        File/folder that should exist inside the repo to verify it.
        Example: "model/train.py"
    """
    repo_dir = Path(project_root) / repo_name

    if marker_file is None:
        marker_path = repo_dir
    else:
        marker_path = repo_dir / marker_file

    if marker_path.exists():
        print(f"{repo_name} repo found: {repo_dir}")
        return repo_dir

    if repo_dir.exists():
        raise FileExistsError(
            f"A folder named {repo_name} already exists, "
            f"but the expected file/folder was not found:\n"
            f"{marker_path}\n"
            "Please delete or rename this folder, then run the script again."
        )

    print(f"{repo_name} repo not found. Cloning...")
    subprocess.run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            repo_url,
            str(repo_dir),
        ],
        cwd=str(project_root),
        check=True,
    )

    if not marker_path.exists():
        raise FileNotFoundError(
            f"{repo_name} was cloned, but expected file/folder was not found:\n"
            f"{marker_path}"
        )

    print(f"{repo_name} repo cloned: {repo_dir}")
    return repo_dir

# Evaluation helpers


_TEST_DATA_CACHE = {}


# Model helpers

def get_best_accuracy_model(model_folder):
    model_path = Path(model_folder) / "best_accuracy.pth"

    if not model_path.exists():
        raise FileNotFoundError(f"best_accuracy.pth not found: {model_path}")

    return model_path


# Test data and evaluation

def get_test_data_for_writer(writer_id, test_samples=TEST_SAMPLES):
    cache_key = (writer_id, test_samples)

    if cache_key not in _TEST_DATA_CACHE:
        test_path = build_real_gobo_data(
            writer_id=writer_id,
            num_samples=test_samples,
            split="test",
        )
        _TEST_DATA_CACHE[cache_key] = test_path

    return _TEST_DATA_CACHE[cache_key]


def evaluate_model_on_writer(
    writer_id,
    model_path,
    test_samples=TEST_SAMPLES,
    show_progress=SHOW_PREDICTION_PROGRESS,
):
    test_path = get_test_data_for_writer(
        writer_id=writer_id,
        test_samples=test_samples,
    )

    results = predict_attentionhtr_folder(
        saved_model_path=model_path,
        image_folder=test_path,
        show_progress=show_progress,
    )

    mean_cer, detailed_results = calculate_cer_from_results(
        results=results,
        gt_path=Path(test_path) / "gt.txt",
    )

    return mean_cer, detailed_results, test_path


# Checkpoint helpers


# Model cleanup

def delete_model_checkpoints(model_path, keep_models=False):
    """
    Deletes large model checkpoint files after evaluation when keep_models=False.

    The result CSV files are not affected.
    Small files such as logs or configuration files remain in the folder.
    """
    if keep_models:
        return []

    model_path = Path(model_path)
    model_folder = model_path.parent

    deleted_files = []

    if not model_folder.exists():
        return deleted_files

    checkpoint_patterns = ["*.pth", "*.pt", "*.ckpt"]

    for pattern in checkpoint_patterns:
        for checkpoint_path in model_folder.glob(pattern):
            try:
                checkpoint_path.unlink()
                deleted_files.append(checkpoint_path)
            except FileNotFoundError:
                pass

    if deleted_files:
        print("Deleted model checkpoints:")
        for deleted_file in deleted_files:
            print(f"  {deleted_file}")

    return deleted_files
