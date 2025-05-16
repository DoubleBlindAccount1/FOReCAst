from abc import ABC, abstractmethod
from typing import List
import numpy as np
from scipy.stats import norm

class Metric(ABC):
    """
    Abstract base class for all evaluation metrics. Defines the interface for computing a metric given predictions and gold values.
    Subclasses must implement the compute method.
    """
    @abstractmethod
    def compute(self, predictions, gold):
        """
        Compute the metric for a set of predictions and gold values.
        Args:
            predictions: Predicted values (list or array).
            gold: Gold/ground-truth values (list or array).
        Returns:
            float: Computed metric value.
        """
        pass

class AccuracyMetric(Metric):
    """
    Computes the accuracy (proportion of correct predictions).
    Suitable for classification tasks.
    """
    def compute(self, predictions, gold):
        """
        Compute accuracy as the proportion of correct predictions.
        Args:
            predictions: List of predicted labels.
            gold: List of gold/ground-truth labels.
        Returns:
            float: Accuracy score (0.0 to 1.0).
        """
        return np.mean([p == g for p, g in zip(predictions, gold)])

class F1ScoreMetric(Metric):
    """
    Computes the F1 score for binary classification, with a configurable positive label.
    """
    def compute(self, predictions, gold, positive_label="yes"):
        """
        Compute the F1 score for binary classification.
        Args:
            predictions: List of predicted labels.
            gold: List of gold/ground-truth labels.
            positive_label (str): The label considered as positive (default: 'yes').
        Returns:
            float: F1 score (0.0 to 1.0).
        """
        tp = sum((p == g == positive_label) for p, g in zip(predictions, gold))
        fp = sum((p == positive_label and g != positive_label) for p, g in zip(predictions, gold))
        fn = sum((p != positive_label and g == positive_label) for p, g in zip(predictions, gold))
        if tp + fp == 0 or tp + fn == 0:
            return 0.0
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

class BrierScoreMetric(Metric):
    """
    Computes the Brier score, a measure of the accuracy of probabilistic predictions.
    Suitable for binary or probabilistic classification.
    """
    def compute(self, pred_probs, gold_outcomes):
        """
        Compute the Brier score as the mean squared error between predicted probabilities and outcomes.
        Args:
            pred_probs: List of predicted probabilities.
            gold_outcomes: List of gold/ground-truth outcomes (0 or 1).
        Returns:
            float: Brier score (lower is better).
        """
        errors = [(p - y) ** 2 for p, y in zip(pred_probs, gold_outcomes)]
        return np.mean(errors)

class CRPSGaussianMetric(Metric):
    """
    Computes the Continuous Ranked Probability Score (CRPS) for Gaussian distributions.
    Suitable for probabilistic regression tasks.
    """
    @staticmethod
    def crps_gaussian(pred, sigma, obs):
        """
        Compute the CRPS for a single Gaussian prediction.
        Args:
            pred: Predicted mean.
            sigma: Predicted standard deviation.
            obs: Observed value.
        Returns:
            float: CRPS value.
        """
        return sigma * (1/np.sqrt(np.pi) - 2 * norm.pdf((obs-pred)/sigma) - (obs-pred)/sigma * (2*norm.cdf((obs-pred)/sigma)-1))

    def compute(self, preds, sigmas, obs):
        """
        Compute the mean CRPS for a set of Gaussian predictions.
        Args:
            preds: List of predicted means.
            sigmas: List of predicted standard deviations.
            obs: List of observed values.
        Returns:
            float: Mean CRPS value.
        """
        return np.mean([self.crps_gaussian(p, s, o) for p, s, o in zip(preds, sigmas, obs)])
