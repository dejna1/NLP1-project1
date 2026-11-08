import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

from models.logreg import LogisticRegressionTextClassifier
from models.mlp import train_neural_network_model, predict_probs, load
from models.random_forest import RandomForestTextClassifier
from text_reps.representations import BagOfWords, TfIdf, TokenIndicator


def build_encoder(representation, tokenizer, fold_training_texts):
    if representation == "token_indicator":
        return TokenIndicator(tokenizer)

    if representation == "bag_of_words":
        return BagOfWords(tokenizer)

    if representation == "tfidf":
        return TfIdf(tokenizer, fold_training_texts)

    raise ValueError(f"Unknown representation: {representation}")


def build_model(architecture, seed, settings):
    if architecture == "logistic_regression":
        return LogisticRegressionTextClassifier(
            random_state=seed,
        )

    if architecture == "random_forest":
        return RandomForestTextClassifier(
            **settings["random_forest"],
            random_state=seed,
        )

    if architecture == "sparse_mlp":
        return None

    raise ValueError(f"Unknown architecture: {architecture}")


def append_result(path, result):
    """
    Adds one cross-validation result to metrics.csv with its
    configuration details.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    frame = pd.DataFrame([result])

    if path.exists():
        frame.to_csv(path, mode="a", header=False, index=False)
    else:
        frame.to_csv(path, index=False)


def run_experiment(train_texts, train_labels, tokenizer, results_csv,
                   models_dir, seeds=(0, 3, 5), num_folds=5,
                   representations=("token_indicator", "bag_of_words", "tfidf"),
                   architectures=("logistic_regression", "sparse_mlp", "random_forest"),
                   num_classes=2, mlp_settings=None, random_forest_settings=None):
    """
    Runs the main experiment of 9 configurations (3 models x 3 tokenizers),
    with the stratified 5-fold-cross-validation, on 3 different seeds.
    Evaluates and saved results in metrics.csv.

    Args:
        train_texts: Train data.
        train_labels: Train labels.
        tokenizer: Built tokenizer, loaded from tokenizer.json.
        results_csv: Path for saving results.
        models_dir: Path for saving models.
        seeds: Defaults to (0, 3, 5), values chosen from the recommended
        example from tutorial.
        num_folds: Defaults to 5.
        representations: Defaults to ("token_indicator", "bag_of_words", "tfidf").
        architectures: Defaults to ("logistic_regression", "sparse_mlp", "random_forest").
        num_classes: Defaults to 2, binary classification.
        mlp_settings: Defaults to None.
        random_forest_settings: Defaults to None.
    """
    mlp_settings = mlp_settings or {}
    random_forest_settings = random_forest_settings or {}

    train_labels = np.asarray(train_labels)

    # loop through the three different seeds to produce independent
    # cross-validation splits - this gives more measurements for
    # the bayesian comparison
    for seed in seeds:
        splitter = StratifiedKFold(n_splits=num_folds,
                                   shuffle=True,
                                   random_state=seed)

        folds = list(splitter.split(train_texts, train_labels))

        for representation in representations:
            for architecture in architectures:
                system_name = f"{representation}__{architecture}"

                for fold_index, (train_indices, validation_indices) in enumerate(folds):
                    fold_train_texts = [train_texts[i] for i in train_indices]
                    fold_train_labels = train_labels[train_indices]

                    fold_validation_texts = [train_texts[i] for i in validation_indices]
                    fold_validation_labels = train_labels[validation_indices]

                    encoder = build_encoder(representation, tokenizer, fold_train_texts)

                    # helps to keep track of the process and debugging
                    start_time = time.time()

                    print(f"[seed={seed} | fold={fold_index + 1}/{num_folds} | "
                          f"{system_name}] training...")

                    # handled on its own as we use the tutorial code which
                    # uses a slightly different architecture than our models
                    if architecture == "sparse_mlp":
                        model_path = (models_dir / f"{system_name}_seed{seed}_fold{fold_index}")

                        _, _, _ = train_neural_network_model(
                            training_set=list(zip(fold_train_texts, fold_train_labels)),
                            validation_set=list(zip(fold_validation_texts, fold_validation_labels)),
                            num_classes=num_classes,
                            encoder=encoder,
                            save_to=model_path,
                            **mlp_settings)

                        trained_mlp = load(model_path, device="cpu")

                        probabilities = predict_probs(fold_validation_texts,
                                                      trained_mlp, encoder)

                        predictions = probabilities.argmax(axis=1)

                    else:
                        model = build_model(architecture, seed,
                                            {"random_forest": random_forest_settings})

                        model.fit(fold_train_texts, fold_train_labels, encoder)

                        probabilities = model.predict_probs(fold_validation_texts,
                                                            encoder)
                        predictions = probabilities.argmax(axis=1)

                        model_path = (models_dir / f"{system_name}_seed{seed}_fold{fold_index}")
                        model.save(model_path)

                    elapsed = time.time() - start_time

                    result = {"representation": representation,
                              "architecture": architecture,
                              "seed": seed,
                              "fold": fold_index,
                              "accuracy": accuracy_score(fold_validation_labels,
                                                         predictions),
                              "micro_f1": f1_score(fold_validation_labels,
                                                   predictions,
                                                   average="micro"),
                              "training_time_seconds": elapsed,
                              "model_path": str(model_path)}

                    append_result(results_csv, result)

                    print(f"[seed={seed} | fold={fold_index + 1}/{num_folds} | "
                          f"{system_name}] "
                          f"accuracy={result['accuracy']:.4f} | "
                          f"micro_f1={result['micro_f1']:.4f} | "
                          f"time={elapsed:.1f}s")

    return pd.read_csv(results_csv)
