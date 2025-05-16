import random
from collections import defaultdict
from typing import List, Dict, Any, Tuple

class DataSplitter:
    """
    A professional, extensible class for splitting a dataset into train, dev, and test sets while maintaining type distribution.
    Provides a static method for flexible, reproducible splits and includes extensive documentation and explicit logic.
    """

    @staticmethod
    def split_dataset(
        data: List[Dict[str, Any]],
        train_ratio: float = 0.65,
        dev_ratio: float = 0.1,
        test_ratio: float = 0.25,
        seed: int = 42
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Split the input dataset into train/dev/test sets, maintaining the distribution of question types.
        Args:
            data (List[Dict[str, Any]]): List of question dictionaries, each with a 'type' and 'id'.
            train_ratio (float): Proportion of data to allocate to the training set.
            dev_ratio (float): Proportion of data to allocate to the development set.
            test_ratio (float): Proportion of data to allocate to the test set.
            seed (int): Random seed for reproducibility.
        Returns:
            Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
                Three lists containing the train, dev, and test splits respectively.
        """
        # Ensure the split ratios sum to 1
        assert abs(train_ratio + dev_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
        # Set the random seed for reproducibility
        random.seed(seed)
        # Bucket the data by question type
        type_buckets = defaultdict(list)
        for entry in data:
            type_buckets[entry["type"]].append(entry)
        train, dev, test = [], [], []
        # Split each bucket according to the specified ratios
        for qtype, entries in type_buckets.items():
            random.shuffle(entries)
            n = len(entries)
            train_end = int(n * train_ratio)
            dev_end = train_end + int(n * dev_ratio)
            train.extend(entries[:train_end])
            dev.extend(entries[train_end:dev_end])
            test.extend(entries[dev_end:])
        # Sort each split by question ID for consistency
        train.sort(key=lambda x: x["id"])
        dev.sort(key=lambda x: x["id"])
        test.sort(key=lambda x: x["id"])
        return train, dev, test
