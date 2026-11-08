from datasets import load_dataset
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from evaluation import run_experiment
from text_reps.tokenization import build_tokenizer, save_tokenizer
from pathlib import Path
from analysis import compare_all_pairs

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
MODELS_DIR = RESULTS_DIR / "models"
METRICS_PATH = RESULTS_DIR / "metrics.csv"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# dataset specific, TweetEval, hate subset
DATASET_NAME = "cardiffnlp/tweet_eval"
LABEL_NAMES = ["non-hate", "hate"]
NUM_CLASSES = len(LABEL_NAMES)

# tutorial 
TOKENIZER_VOCAB_SIZE = 1000
TOKENIZER_MIN_FREQUENCY = 2

# cross-validation configurations
SEEDS = [0, 3, 5]
NUM_FOLDS = 5

# 3 models, 3 representation spaces
REPRESENTATIONS = ["token_indicator", "bag_of_words", "tfidf"]
MODELS = ["logistic_regression", "sparse_mlp", "random_forest"]
MLP_SETTINGS = {
    "hidden_dim": 128,
    "dropout": 0.2,
    "batch_size": 64,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "num_epochs": 20,
}

RANDOM_FOREST_SETTINGS = {
    "n_estimators": 300,
    "class_weight": "balanced",
}


def retrieve_dataset() -> tuple:
    # load ds from HF
    ds = load_dataset("cardiffnlp/tweet_eval", "hate")

    # train: 9000, validation: 1000, test:2970
    X_train = ds['train']['text']
    y_train = ds['train']['label']
    X_validation = ds['validation']['text']
    y_validation = ds['validation']['label']
    X_test = ds['test']['text']
    y_test = ds['test']['label']

    # consider returning only train, test splits - save validation
    # for hyperparameter tuning

    return {"train": (X_train, y_train),
            "validation": (X_validation, y_validation),
            "test": (X_test, y_test)}


def summarise_results(metrics_path):
    metrics = pd.read_csv(metrics_path)

    summary = (metrics.groupby(["representation", "architecture"])
               .agg(mean_accuracy=("accuracy", "mean"),
                    std_accuracy=("accuracy", "std"),
                    mean_micro_f1=("micro_f1", "mean"),
                    std_micro_f1=("micro_f1", "std"),
                    mean_time_seconds=("training_time_seconds", "mean"),
                    num_runs=("accuracy", "count")).reset_index().sort_values("mean_micro_f1", ascending=False))

    summary_path = RESULTS_DIR / "summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\nExperiment summary:")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    return summary


def run_rope_comparisons(metrics_path, rope_size=0.005):
    metrics = pd.read_csv(metrics_path)

    comparisons = compare_all_pairs(metrics, metric="accuracy",
                                    rope_size=rope_size)

    comparisons_path = RESULTS_DIR / "rope_comparisons.csv"
    comparisons.to_csv(comparisons_path, index=False)

    print("\nROPE comparisons:")
    print(comparisons.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    return comparisons


def main():
    """
    Runs the complete TweetEval hate classification experiment.

    The steps are:
    1. Load the TweetEval hate dataset.
    2. Load or train a BPE tokenizer using only the training split.
    3. Run repeated stratified cross-validation for every representation and
       architecture combination.
    4. Summarise the cross-validation results.
    5. Perform Bayesian ROPE comparisons between every pair of systems.
    """
    start_time = time.time()

    print("Loading TweetEval hate dataset...")
    dataset = retrieve_dataset()

    train_texts, train_labels = dataset["train"]

    tokenizer_path = RESULTS_DIR / "tokenizer.json"

    if tokenizer_path.exists():
        print(f"\nLoading existing tokenizer: {tokenizer_path}")
        from text_reps.tokenization import load_tokenizer

        tokenizer = load_tokenizer(tokenizer_path)
    else:
        print("\nTraining BPE tokenizer on the training split only...")
        tokenizer = build_tokenizer(
            train_texts,
            vocab_size=TOKENIZER_VOCAB_SIZE,
            min_frequency=TOKENIZER_MIN_FREQUENCY,
        )
        save_tokenizer(tokenizer, tokenizer_path)

    print(f"Tokenizer vocabulary size: {tokenizer.get_vocab_size()}")

    print("\nStarting cross-validation experiments...")
    metrics = run_experiment(
        train_texts=train_texts,
        train_labels=train_labels,
        tokenizer=tokenizer,
        results_csv=METRICS_PATH,
        models_dir=MODELS_DIR,
        num_classes=NUM_CLASSES,
        mlp_settings=MLP_SETTINGS,
        random_forest_settings=RANDOM_FOREST_SETTINGS,
    )

    summary = summarise_results(METRICS_PATH)
    comparisons = run_rope_comparisons(METRICS_PATH)

    total_time = time.time() - start_time

    print(f"\nAll experiments finished in {total_time / 60:.1f} minutes.")
    print(f"Metrics: {METRICS_PATH}")
    print(f"Summary: {RESULTS_DIR / 'summary.csv'}")
    print(f"ROPE comparisons: {RESULTS_DIR / 'rope_comparisons.csv'}")


if __name__ == "__main__":
    main()
