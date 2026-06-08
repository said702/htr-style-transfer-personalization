"""Configuration for the HTR personalization experiments.

This file controls:
- which experiments are executed,
- which GoBo writers are used,
- where datasets, checkpoints, temporary files, and results are stored,
- whether generated models and LMDB caches are kept after evaluation.

For a full paper reproduction, use all GoBo writers except writer 33.
"""

from pathlib import Path


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_ROOT = PROJECT_ROOT / "data"

REAL_GOBO_ROOT = DATA_ROOT / "real"
SYNTHETIC_ROOT = DATA_ROOT / "synthetic"

# Generic AttentionHTR baseline checkpoint used as the starting point
# for all personalization experiments.
GENERIC_MODEL_PATH = (
    PROJECT_ROOT
    / "generic_model"
    / "AttentionHTR_IAM_baseline_model.pth"
)


# ---------------------------------------------------------------------
# Writer setup
# ---------------------------------------------------------------------

# GoBo contains writers 0-40. Writer 33 is excluded, as in the paper.
WRITER_IDS = [writer_id for writer_id in range(41) if writer_id != 33]



# ---------------------------------------------------------------------
# Experiment control
# ---------------------------------------------------------------------
# Set the experiments you want to run to True.
# All flags are disabled by default to avoid accidentally starting long GPU runs.

# Paper experiment 1:
# Baseline evaluation of the generic AttentionHTR model without personalization.
RUN_BASELINE_EVALUATION = False

# Paper experiment 2:
# Progressive real-data personalization with the target writer's own samples.
RUN_REAL_PERSONALIZATION = False

# Paper experiment 3:
# Random-writer control from the paper: failed "personalization" when using
# samples from a different writer. Can be run alone. If both
# RUN_REAL_PERSONALIZATION and RUN_RANDOM_WRITER_CONTROL are True, the random
# control reuses the models trained during real personalization.
RUN_RANDOM_WRITER_CONTROL = False

# Paper experiment 4:
# Word-group personalization on GoBo.
# This evaluates how different GoBo word groups contribute to personalization.
RUN_WORD_GROUP_PERSONALIZATION = False

# Paper experiment 5a:
# One-shot style-transfer personalization using One-DM synthetic data.
RUN_STYLE_TRANSFER_ONE_SHOT = False

# Paper experiment 5b:
# Few-shot style-transfer personalization using VATr synthetic data.
RUN_STYLE_TRANSFER_FEW_SHOT = False

# Paper experiment 6:
# Mixed real/synthetic personalization.
# This combines real GoBo samples and style-transferred synthetic samples.
RUN_MIXED_REAL_SYNTHETIC = False

# Paper experiment 7:
# Ordered personalization.
# The model is first personalized with real data and then continued with synthetic data.
RUN_ORDERED_REAL_THEN_SYNTHETIC = False

# Paper experiment 8:
# Large-scale progressive style-transfer personalization with up to 10,000 synthetic samples.
RUN_BIG_STYLE_TRANSFER = False

# Paper experiment 9:
# Binarization experiments.
# This evaluates whether non-stylistic image artifacts influence the results.
RUN_BINARIZATION_EXPERIMENT = False


# ---------------------------------------------------------------------
# Training setup
# ---------------------------------------------------------------------

# Personalization training settings used across experiments.
BATCH_SIZE = 10
EPOCHS = 5

# Number of GoBo test samples per writer.
TEST_SAMPLES = 398

RANDOM_SEED = 1


# ---------------------------------------------------------------------
# Progressive real personalization setup
# ---------------------------------------------------------------------

# Progressive real-data personalization setup.
# Used only for RUN_REAL_PERSONALIZATION, i.e. the actual-writer
# personalization and the random-writer control experiments.
BEGIN_SAMPLES = 10

# END_SAMPLES is an upper bound. The final step uses all available
# GoBo training samples if fewer samples are available.
END_SAMPLES = 530

# Step size for adding more writer-specific samples.
STEP_SAMPLES = 10

# Recreate cached progressive folders when the GoBo training-entry filter changes.
# Keep this True if filtering, sampling, or dataset preparation logic changes.
RECREATE_PROGRESSIVE_TRAIN_FOLDERS = True

PROGRESSIVE_TRAIN_ROOT = PROJECT_ROOT / "progressive_real_train"


# ---------------------------------------------------------------------
# Result paths
# ---------------------------------------------------------------------

RESULTS_ROOT = PROJECT_ROOT / "experiment_results"

EVALUATION_BASELINE_RESULTS_CSV = (
    RESULTS_ROOT
    / "Evaluation_Baseline"
    / "Evaluation_Baseline.csv"
)

ACTUAL_RESULTS_CSV = (
    RESULTS_ROOT
    / "personalization_real"
    / "RealData_Actual_Writer"
    / "RealData_Actual_Writer.csv"
)

RANDOM_RESULTS_CSV = (
    RESULTS_ROOT
    / "personalization_real"
    / "RealData_Random_Writer"
    / "RealData_Random_Writer.csv"
)


# ---------------------------------------------------------------------
# Mixed real/synthetic personalization setup
# ---------------------------------------------------------------------

# Real/synthetic ratios used in the mixed real/synthetic experiment.
# Each tuple is (real_ratio, synthetic_ratio).
# Example: (0.80, 0.20) means 80% real samples plus 20% synthetic samples.
MIXED_REAL_SYNTHETIC_RATIOS = [(0.90, 0.10), (0.80, 0.20), (0.50, 0.50)]

MIXED_REAL_SYNTHETIC_DATA_ROOT = PROJECT_ROOT / "mixed_real_synthetic_data"

MIXED_REAL_SYNTHETIC_RESULTS_ROOT = RESULTS_ROOT / "mixed_real_synthetic"


# ---------------------------------------------------------------------
# Ordered real-then-synthetic personalization setup
# ---------------------------------------------------------------------

# Ordered setup:
# first train with real samples, then continue with synthetic samples.
ORDERED_BEGIN_SAMPLES = 50

# Number of real samples used before switching to synthetic data.
ORDERED_REAL_LIMIT = 250

# Upper limit for the ordered experiment.
ORDERED_END_SAMPLES = 530

# Step size for the ordered experiment.
ORDERED_STEP_SAMPLES = 50

ORDERED_REAL_THEN_SYNTHETIC_DATA_ROOT = (
    PROJECT_ROOT / "ordered_real_then_synthetic_data"
)

ORDERED_REAL_THEN_SYNTHETIC_RESULTS_ROOT = (
    RESULTS_ROOT / "ordered_real_then_synthetic"
)

ORDERED_REAL_THEN_SYNTHETIC_RESULTS_CSV = (
    ORDERED_REAL_THEN_SYNTHETIC_RESULTS_ROOT
    / "Ordered_Real_Then_Synthetic.csv"
)


# ---------------------------------------------------------------------
# Style-transfer dataset paths
# ---------------------------------------------------------------------

# Already generated one-shot synthetic GoBo data from Zenodo.
STYLE_TRANSFER_ONE_SHOT_ROOT = (
    SYNTHETIC_ROOT
    / "synthetic_gobo_one_shot"
    / "01_GoBo_synthetic_one_shot"
)

# Already generated few-shot synthetic GoBo data from Zenodo.
STYLE_TRANSFER_FEW_SHOT_ROOT = (
    SYNTHETIC_ROOT
    / "synthetic_gobo_few_shot"
    / "02_GoBo_synthetic_few_shot"
)

# Already generated large-scale few-shot synthetic dataset with 10,000 samples.
BIG_STYLE_TRANSFER_ROOT = (
    SYNTHETIC_ROOT
    / "big_style_transfer"
    / "GoBo_synthetic_few_shot_10k_common_words"
)


# ---------------------------------------------------------------------
# Style-transfer result paths
# ---------------------------------------------------------------------

STYLE_TRANSFER_RESULTS_ROOT = RESULTS_ROOT / "style_transfer"

STYLE_TRANSFER_ONE_SHOT_RESULTS_CSV = (
    STYLE_TRANSFER_RESULTS_ROOT
    / "Style_Transfer_One_Shot"
    / "Style_Transfer_One_Shot.csv"
)

STYLE_TRANSFER_FEW_SHOT_RESULTS_CSV = (
    STYLE_TRANSFER_RESULTS_ROOT
    / "Style_Transfer_Few_Shot"
    / "Style_Transfer_Few_Shot.csv"
)

BIG_STYLE_TRANSFER_RESULTS_CSV = (
    STYLE_TRANSFER_RESULTS_ROOT
    / "Big_Style_Transfer"
    / "Big_Style_Transfer.csv"
)


# ---------------------------------------------------------------------
# Big Style Transfer setup
# ---------------------------------------------------------------------

# Progressive total sample sizes for the 10,000-sample style-transfer experiment.
BIG_STYLE_TRANSFER_BEGIN_SAMPLES = 500
BIG_STYLE_TRANSFER_END_SAMPLES = 10000
BIG_STYLE_TRANSFER_STEP_SAMPLES = 500

# Validation ratio used at each progressive Big Style Transfer step.
BIG_STYLE_TRANSFER_VAL_RATIO = 0.30

# Recreate cached Big Style Transfer train/validation split folders.
# Keep this True if sampling, split settings, or dataset preparation logic changes.
RECREATE_BIG_STYLE_TRANSFER_SPLITS = True

BIG_STYLE_TRANSFER_SPLIT_ROOT = (
    PROJECT_ROOT / "big_style_transfer_train_val_split"
)

BINARIZED_BIG_STYLE_TRANSFER_SPLIT_ROOT = (
    PROJECT_ROOT / "binarized_big_style_transfer_train_val_split"
)


# ---------------------------------------------------------------------
# Word-group personalization setup
# ---------------------------------------------------------------------

WORD_GROUP_DATA_ROOT = PROJECT_ROOT / "word_group_data"

WORD_GROUP_RESULTS_ROOT = RESULTS_ROOT / "word_group_personalization"

WORD_GROUP_RESULTS_CSV = (
    WORD_GROUP_RESULTS_ROOT
    / "word_group_personalization_results.csv"
)

# GoBo word groups used in the word-group personalization experiment.
WORD_GROUP_SELECTED_GROUPS = [
    "brown",
    "cedar",
    "nonwords",
    "domain_specific",
]


# ---------------------------------------------------------------------
# Binarization experiment setup
# ---------------------------------------------------------------------

BINARIZATION_DATA_ROOT = PROJECT_ROOT / "binarization_data"

BINARIZED_STYLE_TRANSFER_FEW_SHOT_ROOT = (
    PROJECT_ROOT
    / "synthetic_gobo_few_shot_binarization"
    / "02_GoBo_synthetic_few_shot"
)

BINARIZED_BIG_STYLE_TRANSFER_ROOT = (
    PROJECT_ROOT
    / "big_style_transfer_binarization"
    / "GoBo_synthetic_few_shot_10k_common_words"
)

BINARIZATION_RESULTS_ROOT = RESULTS_ROOT / "binarization"

STYLE_TRANSFER_BINARIZATION_RESULTS_CSV = (
    BINARIZATION_RESULTS_ROOT
    / "Style_Transfer_Binarization"
    / "Style_Transfer_Binarization.csv"
)

BIG_STYLE_TRANSFER_BINARIZATION_RESULTS_CSV = (
    BINARIZATION_RESULTS_ROOT
    / "Big_Style_Transfer_Binarization"
    / "Big_Style_Transfer_Binarization.csv"
)

# Recreate binarized synthetic data.
# Set to False if the binarized data already exists and should be reused.
RECREATE_BINARIZED_SYNTHETIC_DATA = True


# ---------------------------------------------------------------------
# AttentionHTR paths
# ---------------------------------------------------------------------

# External AttentionHTR repository.
# It is downloaded automatically if missing.
ATTENTION_HTR_ROOT = PROJECT_ROOT / "AttentionHTR"

ATTENTION_HTR_MODEL_ROOT = ATTENTION_HTR_ROOT / "model"

CREATE_LMDB_SCRIPT = ATTENTION_HTR_MODEL_ROOT / "create_lmdb_dataset.py"
TRAIN_SCRIPT = ATTENTION_HTR_MODEL_ROOT / "train.py"


# ---------------------------------------------------------------------
# Temporary model and LMDB paths
# ---------------------------------------------------------------------

# Temporary LMDB training caches created for AttentionHTR.
LMDB_ROOT = PROJECT_ROOT / "lmdb_data"

# Output folder for personalized model checkpoints.
SAVED_MODELS_ROOT = PROJECT_ROOT / "saved_models"


# ---------------------------------------------------------------------
# Storage and cleanup
# ---------------------------------------------------------------------

# To avoid excessive storage usage during long experiment runs,
# model checkpoints are deleted automatically after evaluation.
# Set to True only if you want to keep personalized checkpoints.
KEEP_MODELS = False

# LMDB datasets are temporary AttentionHTR training caches.
# They are deleted automatically after each training run by default.
# Set to True only if you need them for debugging.
KEEP_LMDB = False


# ---------------------------------------------------------------------
# Evaluation setup
# ---------------------------------------------------------------------

# If True, prediction progress is printed during evaluation.
SHOW_PREDICTION_PROGRESS = True


# ---------------------------------------------------------------------
# Image setup
# ---------------------------------------------------------------------

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
