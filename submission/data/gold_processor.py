import math
import json
import hashlib
import unicodedata
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import argparse

class GoldDataProcessor:
    """
    A professional and extensible processor for FOReCAst gold data. This class provides
    methods for reading, cleaning, hashing, and transforming gold data into a format
    suitable for downstream tasks, such as uploading to HuggingFace or further analysis.
    All key steps are documented for clarity and maintainability.
    """

    # Mapping from internal type keys to human-readable types for clarity in output
    type_map = {
        "binary": "Boolean question",
        "numeric": "quantity estimation",
        "date": "timeframe prediction"
    }
    # Set to track used IDs and avoid collisions
    used_ids = set()

    @staticmethod
    def sigmoid(x: float) -> float:
        """
        Compute the sigmoid function for a given input value.
        Args:
            x (float): The input value.
        Returns:
            float: The sigmoid of x.
        """
        return 1 / (1 + math.exp(-x))

    @staticmethod
    def simplify_time(time_string: str) -> str:
        """
        Convert a time string in ISO format to 'YYYY-MM-DD' format for consistency.
        Args:
            time_string (str): The input time string, possibly with a 'Z' timezone.
        Returns:
            str: The formatted date string, or the original value if parsing fails.
        """
        try:
            # Replace 'Z' with '+00:00' for ISO compatibility
            dt = datetime.fromisoformat(time_string.replace("Z", "+00:00"))
            return dt.strftime("%Y-%m-%d")
        except Exception:
            # If parsing fails, return the original string
            return time_string

    @staticmethod
    def generate_hash_id(entry: dict) -> str:
        """
        Generate a short, unique hash ID from the question text and creation time.
        Args:
            entry (dict): The question entry with 'question' and 'created_time'.
        Returns:
            str: An 8-character hash string.
        """
        raw_string = entry["question"] + entry["created_time"]
        hash_digest = hashlib.sha256(raw_string.encode("utf-8")).hexdigest()
        return hash_digest[:8]

    @classmethod
    def get_unique_hashed_id(cls, entry: dict) -> str:
        """
        Ensure that the generated hash ID is unique within the dataset by appending a numeric suffix if needed.
        Args:
            entry (dict): The question entry.
        Returns:
            str: A unique hash ID.
        """
        base_id = cls.generate_hash_id(entry)
        suffix = 0
        unique_id = base_id
        # Loop until a truly unique ID is found
        while unique_id in cls.used_ids:
            suffix += 1
            unique_id = f"{base_id}{suffix}"
        cls.used_ids.add(unique_id)
        return unique_id

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Normalize and clean a text string by removing invisible unicode and standardizing quotes.
        Args:
            text (str): The input string to clean.
        Returns:
            str: The cleaned and normalized string.
        """
        if not isinstance(text, str):
            return text
        # Normalize unicode characters
        normalized = unicodedata.normalize("NFKC", text)
        # Replace curly quotes with straight quotes
        normalized = normalized.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
        # Remove invisible unicode characters (zero-width, BOM, etc.)
        cleaned = re.sub(r"[\u200B-\u200D\uFEFF]", "", normalized)
        return cleaned.strip()

    @staticmethod
    def round_resolution_if_numeric(item: dict) -> None:
        """
        If the item is a quantity estimation and the resolution is numeric, round it to the nearest integer.
        Args:
            item (dict): The question item to process.
        """
        if item.get("type") == "quantity estimation" and "resolution" in item:
            try:
                value = float(item["resolution"])
                item["resolution"] = str(round(value))
            except (ValueError, TypeError):
                # If conversion fails, leave the value unchanged
                pass

    @classmethod
    def process_gold_jsonl(cls, input_path: str, output_path: str) -> None:
        """
        Read a gold JSONL file, clean and transform each entry, and write the processed results to a new file.
        Steps:
            - Skip entries with missing confidence (normalized_score)
            - Assign a new unique hashed ID
            - Rename and round the confidence field
            - Convert type keys to human-readable types
            - Round numeric resolutions for quantity estimation questions
        Args:
            input_path (str): Path to the input JSONL file.
            output_path (str): Path to write the processed JSONL file.
        """
        with open(input_path, encoding="utf-8") as f_in, open(output_path, "w", encoding="utf-8") as f_out:
            for line in f_in:
                # Parse the JSON line into a Python dictionary
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    # Skip lines that are not valid JSON
                    continue
                # Skip if confidence is missing
                if item.get("normalized_score") is None:
                    continue
                # Assign a new unique hashed ID for this entry
                item["id"] = cls.get_unique_hashed_id(item)
                # Rename the confidence field and round to 4 decimals
                item["confidence"] = round(item.pop("normalized_score"), 4)
                # Convert the type field to a human-readable string
                item["type"] = cls.type_map.get(item.get("type"), item["type"])
                # Round the resolution if the question is numeric
                cls.round_resolution_if_numeric(item)
                # Write the processed item as a JSON line to the output file
                f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
                # Normalize and simplify all string fields
                for k, v in item.items():
                    if isinstance(v, str):
                        v = cls.clean_text(v)
                        if "T" in v and ":" in v:
                            v = cls.simplify_time(v)
                        item[k] = v
                # Just in case, normalize question again
                if "question" in item:
                    item["question"] = cls.clean_text(item["question"])
                # Dump to file with unicode preserved
                f_out.write(json.dumps(item, ensure_ascii=False) + '\n')

    @staticmethod
    def process_gold_data(raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Original OOP version for in-memory data
        processed = []
        for entry in raw_data:
            question_data = entry.get("question", {})
            baseline_score = (
                question_data.get("aggregations", {})
                .get("recency_weighted", {})
                .get("score_data", {})
                .get("baseline_score")
            )
            normalized_score = None
            if baseline_score is not None:
                normalized_score = GoldDataProcessor.sigmoid(baseline_score / 100)
            processed.append({
                "id": question_data.get("id"),
                "question": question_data.get("title"),
                "type": entry.get("type"),
                "resolution": question_data.get("resolution"),
                "resolution_time": question_data.get("actual_resolve_time"),
                "created_time": question_data.get("created_at"),
                "normalized_score": normalized_score
            })
        return processed

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare FOReCAst gold data for HuggingFace format.")
    parser.add_argument('--splits', nargs='+', default=['train', 'dev', 'test'], help='Data splits to process')
    parser.add_argument('--suffix', type=str, default='_gold.jsonl', help='Suffix for input files')
    parser.add_argument('--prefix', type=str, default='FOReCAst_', help='Prefix for output files')
    args = parser.parse_args()

    for split in args.splits:
        input_path = f"{split}{args.suffix}"
        output_path = f"{args.prefix}{split}.jsonl"
        GoldDataProcessor.process_gold_jsonl(input_path, output_path)
    print(f"✅ All {args.prefix}{{train,dev,test}}.jsonl files created with cleaned and transformed data.")
