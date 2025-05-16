from abc import ABC, abstractmethod
from typing import List, Dict, Any

class ResultFilter(ABC):
    """
    Abstract base class for all result filters. Defines the interface for filtering a list of records.
    Subclasses must implement the filter method.
    """
    @abstractmethod
    def filter(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter a list of records according to filter-specific logic.
        Args:
            records (List[Dict[str, Any]]): List of record dictionaries.
        Returns:
            List[Dict[str, Any]]: Filtered list of records.
        """
        pass

class CutoffDateFilter(ResultFilter):
    """
    Filters records to only those matching a specific cutoff date for each model.
    Useful for evaluating model performance at a specific snapshot in time.
    """
    def __init__(self, cutoff_dates: Dict[str, str]):
        """
        Initialize the filter with a mapping of model names to cutoff dates.
        Args:
            cutoff_dates (Dict[str, str]): Mapping from model short name to cutoff date string.
        """
        self.cutoff_dates = cutoff_dates

    def filter(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter records to only those with a filter_date matching the expected cutoff for their model.
        Args:
            records (List[Dict[str, Any]]): List of record dictionaries.
        Returns:
            List[Dict[str, Any]]: Filtered list of records.
        """
        filtered = []
        for record in records:
            model = record.get('model')
            filter_date = record.get('filter_date')
            short_model = self._get_short_model(model)
            expected_date = self.cutoff_dates.get(short_model)
            if expected_date and filter_date == expected_date:
                filtered.append(record)
        return filtered

    def _get_short_model(self, model_str: str) -> str:
        """
        Extract the short model name from a model string by matching suffixes in the cutoff_dates keys.
        Args:
            model_str (str): Full model string.
        Returns:
            str: Short model name if found, else None.
        """
        for key in self.cutoff_dates:
            if model_str and model_str.endswith(key):
                return key
        return None

class ModelTypeFilter(ResultFilter):
    """
    Filters records to only those matching a specific model type substring.
    Useful for evaluating a subset of models (e.g., only 'instruct' models).
    """
    def __init__(self, model_type: str):
        """
        Initialize the filter with a model type substring (case-insensitive).
        Args:
            model_type (str): Substring to match in the model field.
        """
        self.model_type = model_type

    def filter(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter records to only those whose model field contains the model_type substring.
        Args:
            records (List[Dict[str, Any]]): List of record dictionaries.
        Returns:
            List[Dict[str, Any]]: Filtered list of records.
        """
        return [r for r in records if self.model_type in r.get('model', '').lower()]
