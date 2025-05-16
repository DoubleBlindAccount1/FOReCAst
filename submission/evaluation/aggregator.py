from abc import ABC, abstractmethod
from typing import List, Dict, Any
import numpy as np

class Aggregator(ABC):
    """
    Abstract base class for all aggregation strategies. Defines the interface for aggregating a list of model outputs.
    Subclasses must implement the aggregate method.
    """
    @abstractmethod
    def aggregate(self, outputs: List[Dict[str, Any]]) -> Any:
        """
        Aggregate a list of model outputs into a single result.
        Args:
            outputs (List[Dict[str, Any]]): List of output dictionaries, each with at least a 'value' key.
        Returns:
            Any: Aggregated result (type depends on the subclass implementation).
        """
        pass

class MajorityVoteAggregator(Aggregator):
    """
    Aggregator that selects the value occurring most frequently among the outputs (majority vote).
    Suitable for categorical or binary outputs.
    """
    def aggregate(self, outputs: List[Dict[str, Any]]) -> Any:
        """
        Perform majority voting over the 'value' field of the outputs.
        Args:
            outputs (List[Dict[str, Any]]): List of output dictionaries.
        Returns:
            Any: The value with the highest count, or None if no values are present.
        """
        values = [o["value"] for o in outputs]
        if not values:
            return None
        # Find the value with the highest frequency
        return max(set(values), key=values.count)

class HighestConfidenceAggregator(Aggregator):
    """
    Aggregator that selects the output with the highest confidence score.
    Suitable for outputs that include a 'confidence' field.
    """
    def aggregate(self, outputs: List[Dict[str, Any]]) -> Any:
        """
        Select the output with the highest 'confidence' value.
        Args:
            outputs (List[Dict[str, Any]]): List of output dictionaries.
        Returns:
            Any: The output dictionary with the highest confidence, or None if the list is empty.
        """
        if not outputs:
            return None
        # Find the output with the highest confidence score
        return max(outputs, key=lambda x: x.get("confidence", 0))

class WeightedAverageAggregator(Aggregator):
    """
    Aggregator that computes the weighted average of numeric values, using the 'confidence' field as the weight.
    Suitable for regression or numeric prediction tasks.
    """
    def aggregate(self, outputs: List[Dict[str, Any]]) -> Any:
        """
        Compute the weighted average of the 'value' fields, weighted by 'confidence'.
        Args:
            outputs (List[Dict[str, Any]]): List of output dictionaries.
        Returns:
            Any: Weighted average of values, or None if not computable.
        """
        values, weights = [], []
        for o in outputs:
            try:
                values.append(float(o["value"]))
                weights.append(float(o.get("confidence", 1.0)))
            except Exception:
                # Skip outputs that cannot be converted to float
                continue
        if not values or not weights or sum(weights) == 0:
            return None
        # Compute the weighted average using numpy
        return np.average(values, weights=weights)
