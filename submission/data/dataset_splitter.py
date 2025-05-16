import json
import random
from collections import defaultdict

class DatasetSplitter:
    """
    A professional, extensible class for splitting datasets into train/dev/test sets while maintaining type distribution.
    Provides detailed logging, extensive comments, and explicit logic for maintainability and clarity.
    """
    def __init__(self, input_file: str = "questions.jsonl", train_ratio: float = 0.65, dev_ratio: float = 0.1, test_ratio: float = 0.25, seed: int = 42):
        """
        Initialize the DatasetSplitter.
        Args:
            input_file (str): Path to the input JSONL file containing the dataset.
            train_ratio (float): Proportion of data to allocate to the training set.
            dev_ratio (float): Proportion of data to allocate to the development set.
            test_ratio (float): Proportion of data to allocate to the test set.
            seed (int): Random seed for reproducibility.
        """
        self.input_file = input_file
        self.train_ratio = train_ratio
        self.dev_ratio = dev_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        # Load the dataset from the input file
        self.data = self.load_jsonl(input_file)

    @staticmethod
    def load_jsonl(filename: str) -> list:
        """
        Load a JSONL file into a list of dictionaries.
        Args:
            filename (str): Path to the JSONL file.
        Returns:
            list: List of dictionary entries from the file.
        """
        with open(filename, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    @staticmethod
    def save_jsonl(filename: str, data: list) -> None:
        """
        Save a list of dictionaries to a JSONL file.
        Args:
            filename (str): Path to the output JSONL file.
            data (list): List of dictionaries to write.
        """
        with open(filename, "w", encoding="utf-8") as f:
            for entry in data:
                f.write(json.dumps(entry) + "\n")

    def split_dataset(self) -> tuple:
        """
        Split the loaded dataset into train/dev/test sets, maintaining type distribution.
        Returns:
            tuple: (train, dev, test) lists of dictionaries.
        """
        # Ensure that the split ratios sum to 1
        assert abs(self.train_ratio + self.dev_ratio + self.test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
        # Set the random seed for reproducibility
        random.seed(self.seed)
        # Bucket the data by question type
        type_buckets = defaultdict(list)
        for entry in self.data:
            type_buckets[entry["type"]].append(entry)
        train, dev, test = [], [], []
        # Split each type bucket according to the specified ratios
        for qtype, entries in type_buckets.items():
            random.shuffle(entries)
            n = len(entries)
            train_end = int(n * self.train_ratio)
            dev_end = train_end + int(n * self.dev_ratio)
            train.extend(entries[:train_end])
            dev.extend(entries[train_end:dev_end])
            test.extend(entries[dev_end:])
        # Sort each split by question ID for consistency
        train.sort(key=lambda x: x["id"])
        dev.sort(key=lambda x: x["id"])
        test.sort(key=lambda x: x["id"])
        return train, dev, test

    def save_splits(self, train_file: str = "train.jsonl", dev_file: str = "dev.jsonl", test_file: str = "test.jsonl") -> None:
        """
        Save the train/dev/test splits to their respective files.
        Args:
            train_file (str): Output file for the training set.
            dev_file (str): Output file for the development set.
            test_file (str): Output file for the test set.
        """
        train, dev, test = self.split_dataset()
        self.save_jsonl(train_file, train)
        self.save_jsonl(dev_file, dev)
        self.save_jsonl(test_file, test)
        print(f"Dataset split into train: {len(train)}, dev: {len(dev)}, test: {len(test)}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Split dataset into train/dev/test sets while maintaining type distribution.")
    parser.add_argument('--input_file', type=str, default="questions.jsonl", help='Input JSONL file')
    parser.add_argument('--train_ratio', type=float, default=0.65, help='Train split ratio')
    parser.add_argument('--dev_ratio', type=float, default=0.1, help='Dev split ratio')
    parser.add_argument('--test_ratio', type=float, default=0.25, help='Test split ratio')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--train_file', type=str, default="train.jsonl", help='Output train JSONL file')
    parser.add_argument('--dev_file', type=str, default="dev.jsonl", help='Output dev JSONL file')
    parser.add_argument('--test_file', type=str, default="test.jsonl", help='Output test JSONL file')
    args = parser.parse_args()
    splitter = DatasetSplitter(
        input_file=args.input_file,
        train_ratio=args.train_ratio,
        dev_ratio=args.dev_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )
    splitter.save_splits(args.train_file, args.dev_file, args.test_file)
