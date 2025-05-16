import os
import json

class GoldSplitter:
    """
    A professional, extensible class for splitting a gold standard dataset into train, dev, and test sets based on IDs from split files.
    Designed for clarity, maintainability, and extensibility. Equivalent to the logic in split_gold.py, but as a reusable class.
    """

    @staticmethod
    def load_ids(file_path: str) -> set:
        """
        Load a set of IDs from a JSONL file. Each line must contain a dictionary with an 'id' key.
        Args:
            file_path (str): Path to the split file (train/dev/test JSONL).
        Returns:
            set: Set of IDs loaded from the file.
        """
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found, skipping.")
            return set()
        ids = set()
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    ids.add(data["id"])
                except json.JSONDecodeError:
                    print(f"Error decoding JSON in {file_path}")
        return ids

    def __init__(self, gold_file: str = "gold.jsonl", splits_dir: str = "."):
        """
        Initialize the GoldSplitter.
        Args:
            gold_file (str): Path to the gold.jsonl file containing all gold data.
            splits_dir (str): Directory containing train/dev/test split files.
        """
        self.gold_file = gold_file
        self.splits_dir = splits_dir

    def split_gold_data(self) -> None:
        """
        Split the gold dataset into train_gold.jsonl, dev_gold.jsonl, and test_gold.jsonl based on IDs from split files.
        Writes sorted output files for each split.
        """
        # Load sets of IDs for each split
        train_ids = self.load_ids(os.path.join(self.splits_dir, "train.jsonl"))
        dev_ids = self.load_ids(os.path.join(self.splits_dir, "dev.jsonl"))
        test_ids = self.load_ids(os.path.join(self.splits_dir, "test.jsonl"))
        split_data = {"train": [], "dev": [], "test": []}
        # Read the gold file and assign each question to the correct split
        with open(self.gold_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    question_id = data["id"]
                    if question_id in train_ids:
                        split_data["train"].append(data)
                    elif question_id in dev_ids:
                        split_data["dev"].append(data)
                    elif question_id in test_ids:
                        split_data["test"].append(data)
                except json.JSONDecodeError:
                    print(f"Error decoding JSON in {self.gold_file}")
        # Sort each split by question ID for consistency
        for key in split_data:
            split_data[key].sort(key=lambda x: x["id"])
        # Write each split to its own output file
        for key in split_data:
            output_path = os.path.join(self.splits_dir, f"{key}_gold.jsonl")
            with open(output_path, "w", encoding="utf-8") as f:
                for item in split_data[key]:
                    f.write(json.dumps(item) + "\n")
            print(f"✅ {output_path} saved with sorted IDs.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Split gold.jsonl into train_gold.jsonl, dev_gold.jsonl, and test_gold.jsonl based on IDs from split files.")
    parser.add_argument('--gold_file', type=str, default="gold.jsonl", help='Path to gold.jsonl')
    parser.add_argument('--splits_dir', type=str, default=".", help='Directory containing train/dev/test split files')
    args = parser.parse_args()
    splitter = GoldSplitter(args.gold_file, args.splits_dir)
    splitter.split_gold_data()
