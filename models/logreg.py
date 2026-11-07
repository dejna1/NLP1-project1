import numpy as np


class LogisticRegression():
    """
    Logistic Regression classifier for binary classification.

    Implementation includes gradient descent optimization with validation monitoring
    and binary cross-entropy loss tracking.

    Attributes:
        learning_rate: Learning rate for gradient descent
        n_iterations: Number of training iterations
        val_observations: Validation features for monitoring
        val_ground_truth: Validation labels for monitoring
        train_loss: List storing training loss over iterations
        val_loss: List storing validation loss over iterations
        parameters: Dictionary storing model parameters (weights, bias)
    """

    def __init__(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        learning_rate=0.01,
        n_iterations=2000,
    ):
        """
        Initialize Logistic Regression model.

        Args:
            X_val: Validation features for monitoring during training
            y_val: Validation labels for monitoring during training
            learning_rate: Learning rate for gradient descent (default: 0.01)
            n_iterations: Number of training iterations (default: 2000)

        Raises:
            ValueError: If X_val and y_val have incompatible shapes
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.val_observations = X_val
        self.val_ground_truth = y_val
        self.train_loss = []
        self.val_loss = []
        # store weights and bias
        self.parameters = {}

    def _sigmoid(self, z: np.ndarray) -> float:
        """
        Sigmoid activation function.

        Converts input values to probabilities between 0 and 1.

        Args:
            z: Input values (numpy array or scalar)

        Returns:
            Probabilities in range (0, 1)
        """
        return 1 / (1 + np.exp(-z))

    def _log_loss(self, y_true, y_pred):
        """
        Compute binary cross-entropy loss.

        Args:
            y_true: True labels
            y_pred: Predicted probabilities

        Returns:
            Binary cross-entropy loss value
        """
        eps = 1e-8
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(
            y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
        )

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Train the logistic regression model using gradient descent.

        Args:
            observations: Training features (numpy array)
            ground_truth: Training labels (numpy array)

        Side Effects:
            - Updates model weights and bias
            - Tracks training and validation loss over iterations
            - Stores final parameters in self.parameters

        Raises:
            ValueError: If observations and ground_truth have incompatible shapes
        """
        # retrieve dimensions
        n_samples, n_features = observations.shape

        # initialize weight and bias
        self.weight = np.zeros(n_features)
        self.bias = 0.0

        # implemented gradient descent
        for _ in range(self.n_iterations):
            linear_output = np.dot(observations, self.weight) + self.bias
            predictions = self._sigmoid(linear_output)

            # gradients
            error = predictions - ground_truth
            gradient_weight = (1 / n_samples) * np.dot(observations.T, error)
            gradient_bias = (1 / n_samples) * np.sum(error)

            # update
            self.weight -= self.learning_rate * gradient_weight
            self.bias -= self.learning_rate * gradient_bias

            self.train_loss.append(self._log_loss(ground_truth, predictions))

            val_logits = np.dot(self.val_observations, self.weight) + self.bias
            val_pred = self._sigmoid(val_logits)
            self.val_loss.append(
                self._log_loss(self.val_ground_truth, val_pred)
            )

        self.parameters = {"weights": self.weight, "bias": self.bias}

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Make binary predictions using the trained model.

        Args:
            observations: Input features for prediction (numpy array)

        Returns:
            Binary predictions (0 or 1) as numpy array

        Raises:
            RuntimeError: If model has not been trained (weights not initialized)
        """
        linear_output = np.dot(observations, self.weight) + self.bias
        probs = self._sigmoid(linear_output)
        return (probs >= 0.5).astype(int)

    def predict_proba(self, observations: np.ndarray) -> np.ndarray:
        """
        Return probability predictions for logistic regression.

        Args:
            observations: Input features for probability prediction (numpy array)

        Returns:
            Probability array where each row contains [P(class_0), P(class_1)]
        """
        linear_output = np.dot(observations, self.weight) + self.bias
        probs = self._sigmoid(linear_output)
        return np.vstack((1 - probs, probs)).T
