import pickle
from pathlib import Path
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression


class LogisticRegressionTextClassifier:
    """
    Logistic Regression classifier for binary classification.

    Uses scikit-learn's LogisticRegression.
    """

    def __init__(self, solver="saga", max_iter=1000, class_weight="balanced",
                 random_state=0):
        """
        Initializes LogReg.

        Args:
            solver (str): Defaults to "saga", same as in the tutorial.
            max_iter (int): Defaults to 1000.
            class_weight (str): Defaults to "balanced".
            random_state (int): Defaults to 0, as solver defaults to 'saga'.
        """
        self.solver = solver
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.random_state = random_state
        self.model = None

    def _encode(self, texts, encoder):
        """
        Encode text using the encoder/tokenizer being used.
        """
        return csr_matrix([encoder(text) for text in texts])

    def fit(self, texts, labels, encoder):
        X = self._encode(texts, encoder)

        self.model = LogisticRegression(solver=self.solver,
                                        max_iter=self.max_iter,
                                        class_weight=self.class_weight,
                                        random_state=self.random_state)
        self.model.fit(X, labels)
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
