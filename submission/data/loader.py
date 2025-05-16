from abc import ABC, abstractmethod
from typing import Any, List
import json

class BaseDataLoader(ABC):
    """
    Abstract base class for all data loaders. Defines the interface for loading data from a given path.
    Subclasses must implement the load method.
    """

    @abstractmethod
    def load(self, path: str) -> List[Any]:
        """
        Load data from the specified path.
        Args:
            path (str): Path to the data file.
        Returns:
            List[Any]: Loaded data as a list of objects.
        """
        pass

class JsonlDataLoader(BaseDataLoader):
    """
    Data loader for JSON Lines (JSONL) files. Each line in the file should be a valid JSON object.
    Handles decoding errors gracefully and returns a list of parsed objects.
    """
    def load(self, path: str) -> List[Any]:
        """
        Load data from a JSONL file, returning a list of dictionaries/objects.
        Args:
            path (str): Path to the JSONL file.
        Returns:
            List[Any]: List of parsed JSON objects.
        """
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    # Attempt to parse each line as JSON
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    # Skip lines that are not valid JSON
                    continue
        return data
