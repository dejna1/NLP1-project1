import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier


class RandomForestTextClassifier: 
    def __init__(self, n_estimators=300, class_weight="balanced", random_state=0,
                 n_jobs=-1):
        """
        Initializes Random Forest classifier.

        Args:
            n_estimators: Number of trees. Defaults to 300.
            class_weight: Defaults to "balanced".
            random_state: Defaults to 0.
            n_jobs: Defaults to -1, value reused form the tutorial.
        """
        self.n_estimators = n_estimators
        self.class_weight = class_weight
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.model = None

    def _encode(self, texts, encoder):
        """
        Encode text using the encoder/tokenizer being used.
        """
        return np.vstack([
            np.asarray(encoder(text), dtype=np.float32).ravel()
            for text in texts
        ])

    def fit(self, texts, labels, encoder):
        features = self._encode(texts, encoder)

        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            class_weight=self.class_weight,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )

        self.model.fit(features, labels)
        return self

    def predict(self, texts, encoder):
        return self.model.predict(self._encode(texts, encoder))

    def predict_probs(self, texts, encoder):
        return self.model.predict_proba(self._encode(texts, encoder))

    # saving and loading trained model for easier evaluation and future uses
    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(f"{path}.pickle", "wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, path):
        with open(f"{path}.pickle", "rb") as file:
            return pickle.load(file)
