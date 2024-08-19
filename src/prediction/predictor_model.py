import os
import warnings
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.exceptions import NotFittedError
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore")


PREDICTOR_FILE_NAME = "predictor.joblib"


class Classifier:
    """A wrapper class for the Support Vector binary classifier."""

    model_name = "Support Vector Binary Classifier"

    def __init__(
        self,
        C: Optional[float] = 1.0,
        kernel: Optional[str] = "rbf",
        degree: Optional[int] = 3,
        decision_threshold: Optional[float] = 0.5,
        positive_class_weight: Optional[float] = 1.0,
        **kwargs,
    ):
        """Construct a new binary classifier.

        Args:
            C (float, optional): Inverse of regularization strength; must be a positive
                float. Like in support vector machines, smaller values specify
                stronger regularization. Must be strictly positive (C >= 0.)
                Defaults to 1.0.
            kernel (int, optional): Specifies the kernel type to be used
                in the algorithm.
                Options to consider: ["linear", "poly", "rbf"]
                Defaults to "rbf".
            degree (int, optional): Degree of the polynomial kernel
                function (poly). This argument is ignored for all other kernel
                functions. Must be >= 1.
                Defaults to 3.
            decision_threshold (float, optional): The decision threshold for
                the positive class. Defaults to 0.5.
            positive_class_weight (float, optional): The weight of the positive
                class. Defaults to 1.0.
        """
        self.C = float(C)
        self.kernel = kernel
        self.degree = int(degree)
        self.decision_threshold = float(decision_threshold)
        self.positive_class_weight = float(positive_class_weight)
        self.model = self.build_model()
        self._is_trained = False

    def build_model(self) -> SVC:
        """Build a new binary classifier."""
        model = SVC(
            C=self.C,
            kernel=self.kernel,
            degree=self.degree,
            probability=True,
            class_weight={0: 1.0, 1: self.positive_class_weight},
        )
        return model

    def fit(self, train_inputs: pd.DataFrame, train_targets: pd.Series) -> None:
        """Fit the binary classifier to the training data.

        Args:
            train_inputs (pandas.DataFrame): The features of the training data.
            train_targets (pandas.Series): The labels of the training data.
        """
        self.model.fit(train_inputs, train_targets)
        self._is_trained = True

    def predict(
        self,
        inputs: pd.DataFrame,
        decision_threshold: float = -1,
    ) -> np.ndarray:
        """Predict class labels for the given data.

        Args:
            inputs (pandas.DataFrame): The input data.
            decision_threshold (Optional float): Decision threshold for the
                positive class.
                Value -1 indicates use the default set when model was
                instantiated.
        Returns:
            numpy.ndarray: The predicted class labels.
        """
        if decision_threshold == -1:
            decision_threshold = self.decision_threshold
        if self.model is not None:
            prob = self.predict_proba(inputs)
            labels = prob[:, 1] >= decision_threshold
        return labels

    def predict_proba(self, inputs: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities for the given data.

        Args:
            inputs (pandas.DataFrame): The input data.
        Returns:
            numpy.ndarray: The predicted class probabilities.
        """
        return self.model.predict_proba(inputs)

    def evaluate(
        self,
        test_inputs: pd.DataFrame,
        test_targets: pd.Series,
        decision_threshold: float = -1,
    ) -> float:
        """Evaluate the classifier and return the accuracy.

        Args:
            test_inputs (pandas.DataFrame): The features of the test data.
            test_targets (pandas.Series): The labels of the test data.
            decision_threshold (Optional float): Decision threshold for the
                positive class.
                Value -1 indicates use the default set when model was
                instantiated.
        Returns:
            float: The accuracy of the classifier.
        """
        if decision_threshold == -1:
            decision_threshold = self.decision_threshold
        if self.model is not None:
            prob = self.predict_proba(test_inputs)
            labels = prob[:, 1] >= decision_threshold
            score = f1_score(test_targets, labels)
            return score

        raise NotFittedError("Model is not fitted yet.")

    def save(self, model_dir_path: str) -> None:
        """Save the binary classifier to disk.

        Args:
            model_dir_path (str): Dir path to which to save the model.
        """
        if not self._is_trained:
            raise NotFittedError("Model is not fitted yet.")
        joblib.dump(self, os.path.join(model_dir_path, PREDICTOR_FILE_NAME))

    @classmethod
    def load(cls, model_dir_path: str) -> "Classifier":
        """Load the binary classifier from disk.

        Args:
            model_dir_path (str): Dir path to the saved model.
        Returns:
            Classifier: A new instance of the loaded binary classifier.
        """
        model = joblib.load(os.path.join(model_dir_path, PREDICTOR_FILE_NAME))
        return model

    def __str__(self):
        return (
            f"Model name: {self.model_name} ("
            f"C: {self.C}, "
            f"degree: {self.degree}, "
            f"kernel: {self.kernel})"
        )


def train_predictor_model(
    train_inputs: pd.DataFrame, train_targets: pd.Series, hyperparameters: dict
) -> Classifier:
    """
    Instantiate and train the predictor model.

    Args:
        train_X (pd.DataFrame): The training data inputs.
        train_y (pd.Series): The training data labels.
        hyperparameters (dict): Hyperparameters for the classifier.

    Returns:
        'Classifier': The classifier model
    """
    classifier = Classifier(**hyperparameters)
    classifier.fit(train_inputs=train_inputs, train_targets=train_targets)
    return classifier


def predict_with_model(
    classifier: Classifier, data: pd.DataFrame, return_probs=False
) -> np.ndarray:
    """
    Predict class probabilities for the given data.

    Args:
        classifier (Classifier): The classifier model.
        data (pd.DataFrame): The input data.
        return_probs (bool): Whether to return class probabilities or labels.
            Defaults to True.

    Returns:
        np.ndarray: The predicted classes or class probabilities.
    """
    if return_probs:
        return classifier.predict_proba(data)
    return classifier.predict(data)


def save_predictor_model(model: Classifier, predictor_dir_path: str) -> None:
    """
    Save the classifier model to disk.

    Args:
        model (Classifier): The classifier model to save.
        predictor_dir_path (str): Dir path to which to save the model.
    """
    if not os.path.exists(predictor_dir_path):
        os.makedirs(predictor_dir_path)
    model.save(predictor_dir_path)


def load_predictor_model(predictor_dir_path: str) -> Classifier:
    """
    Load the classifier model from disk.

    Args:
        predictor_dir_path (str): Dir path where model is saved.

    Returns:
        Classifier: A new instance of the loaded classifier model.
    """
    return Classifier.load(predictor_dir_path)


def evaluate_predictor_model(
    model: Classifier,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    decision_threshold: float = -1,
) -> float:
    """
    Evaluate the classifier model and return the accuracy.

    Args:
        model (Classifier): The classifier model.
        x_test (pd.DataFrame): The features of the test data.
        y_test (pd.Series): The labels of the test data.
        decision_threshold (Union(optional, float)): Decision threshold
                for predicted label.
                Value -1 indicates use the default set when model was
                instantiated.

    Returns:
        float: The accuracy of the classifier model.
    """
    return model.evaluate(x_test, y_test, decision_threshold)


def set_decision_threshold(model: Classifier, decision_threshold: float) -> None:
    """
    Set the decision threshold for the classifier model.

    Args:
        model (Classifier): The classifier model.
        decision_threshold (float): The decision threshold.
    """
    model.decision_threshold = decision_threshold
