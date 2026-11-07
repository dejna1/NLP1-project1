import numpy as np
from sklearn.ensemble import RandomForestClassifier


class RandomForestTextClassifier:
    def __init__(self, encoder, n_estimators=300, max_depth=None,
                 min_samples_split=2, min_samples_leaf=1, class_weight=None):
        """
        Initializes the RandomForest model.

        Args:
            encoder: Chosen tokenizer.
            n_estimators: Number of trees in the forest.
            max_depth: Defaults to None, nodes are expanded until all
            leaves contain less than min_samples_split samples or all leaves
            are pure.
            class_weight: Defaults to None - maybe edit with the offensive
            dataset label distribution.
        """
        self.encoder = encoder
        self.model = RandomForestClassifier(n_estimators=n_estimators,
                                            max_depth=max_depth,
                                            min_samples_split=min_samples_split,
                                            min_samples_leaf=min_samples_leaf,
                                            class_weight=class_weight)

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, epochs: int):
        """
        Trains the Random Forest model.

        Args:
            x_train (np.ndarray): Training data.
            y_train (np.ndarray): Training labels.
            epochs (int): Number of training epochs.
        """
        X = self._encode(x_train)
        self.model.fit(X, y_train, epochs=epochs)
        return self

    def predict(self, x_test: np.ndarray):
        """
        Evaluates the trained model on test data.

        Args:
            x_test (np.ndarray): Test data.
        """
        X = self._encode(x_test)
        return self.model.predict(X)

    def predict_proba(self, texts: np.ndarray):
        """
        (same as the tutorial)
        Predicts labels (via the most probable class decision rule) for a collection of texts. 
        (scikit-learn hard-codes the most probable class as the decision rule of choice)
        
        texts: a list of texts, each a string
        encoder: the Encoder used for in the model's training
        """
        X = self._encode(texts)
        return self.model.predict_proba(X)

    def _encode(self, texts: np.ndarray, labels: np.ndarray):
        """
        Encodes texts using the chosen tokenizer.

        Args:
            texts (np.ndarray): Data.
            labels (np.ndarray): Labels.
        """
        return [self.encoder(text) for text, label in texts]