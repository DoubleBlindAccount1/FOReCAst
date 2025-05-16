import json
import re
import glob
import numpy as np
from collections import Counter
from word2number import w2n

class AnswerPostprocessor:
    """
    A professional, extensible postprocessor for cleaning, aggregating, and postprocessing model answer outputs in .jsonl files.
    This class provides robust methods for cleaning binary, numeric, and date answers, as well as aggregation utilities.
    Every method is documented for clarity and future maintainability.
    """
    # Mapping of various affirmative/negative expressions to canonical binary answers
    BINARY_MAP = {
        "yes": "yes", "no": "no", "true": "yes", "false": "no",
        "sure": "yes", "absolutely": "yes", "certainly": "yes", "correct": "yes",
        "of course": "yes", "indeed": "yes", "affirmative": "yes",
        "not at all": "no", "unlikely": "no", "no way": "no",
        "never": "no", "impossible": "no", "incorrect": "no"
    }

    @classmethod
    def clean_binary(cls, value: str) -> str:
        """
        Clean and normalize binary (yes/no) answers.
        Args:
            value (str): The answer string to clean.
        Returns:
            str: Canonical binary answer ('yes', 'no', or 'unclear').
        """
        value = value.lower().strip()
        for word, mapped in cls.BINARY_MAP.items():
            if word in value:
                return mapped
        return "unclear"

    @staticmethod
    def clean_numeric(value: str) -> str:
        """
        Clean and extract numeric values from the answer string.
        Args:
            value (str): The answer string to clean.
        Returns:
            str: Extracted numeric value as a string, or 'unclear'.
        """
        value = value.lower().strip()
        # Try to extract a number using regex
        match = re.search(r"-?\d+(\.\d+)?", value)
        if match:
            return match.group()
        # Try to convert words to numbers
        words = value.split()
        for word in words:
            try:
                num = w2n.word_to_num(word)
                return str(num)
            except ValueError:
                continue
        return "unclear"

    @staticmethod
    def clean_date(value: str) -> str:
        """
        Clean and extract date values from the answer string.
        Args:
            value (str): The answer string to clean.
        Returns:
            str: Extracted date in ISO format (YYYY-MM-DD), or 'unclear'.
        """
        value = value.lower().strip()
        # Try to match YYYY-MM-DD
        match = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", value)
        if match:
            return match.group()
        # Try to match Month YYYY
        match = re.search(r"\b(january|february|march|april|may|june|july|august|september|october|november|december) (\d{4})\b", value)
        if match:
            return f"{match.group(2)}-{match.group(1)[:3]}-01"
        # Try to match just a year
        match = re.search(r"\b(19\d{2}|20\d{2})\b", value)
        if match:
            return f"{match.group()}-01-01"
        return "unclear"

    @classmethod
    def clean_output(cls, value: str, q_type: str) -> str:
        """
        Clean an answer string based on its question type.
        Args:
            value (str): The answer string to clean.
            q_type (str): The type of question ('binary', 'numeric', or 'date').
        Returns:
            str: Cleaned answer or 'unclear'.
        """
        if not isinstance(value, str) or not value.strip():
            return "unclear"
        value = value.strip().lower()
        # Remove trailing punctuation and split on newlines, commas, periods, or closing braces
        value = re.split(r'[\n,.\}]', value)[0].strip()
        if q_type == "binary":
            return cls.clean_binary(value)
        elif q_type == "numeric":
            return cls.clean_numeric(value)
        elif q_type == "date":
            return cls.clean_date(value)
        return "unclear"

    @staticmethod
    def majority_vote(outputs: list) -> tuple:
        """
        Perform a majority vote over a list of outputs, returning the most common value and its highest confidence.
        Args:
            outputs (list): List of output dicts with 'value' and 'confidences'.
        Returns:
            tuple: (most common value, highest confidence for that value)
        """
        counts = Counter(o["value"] for o in outputs)
        most_common = counts.most_common(1)[0][0]
        conf = max(o["confidences"]["normalized_confidences"] for o in outputs if o["value"] == most_common)
        return most_common, conf

    @staticmethod
    def highest_confidence(outputs: list) -> dict:
        """
        Select the output with the highest normalized confidence.
        Args:
            outputs (list): List of output dicts with 'confidences'.
        Returns:
            dict: Output dict with the highest normalized confidence.
        """
        best = max(outputs, key=lambda x: x["confidences"]["normalized_confidences"])
        return best
        return best["value"], best["confidences"]["normalized_confidences"]

    @staticmethod
    def weighted_average(outputs):
        counts = Counter(o["value"] for o in outputs)
        most_common = counts.most_common(1)[0][0]
        weights = [o["confidences"]["avg_confidences"] for o in outputs if o["value"] == most_common]
        return most_common, np.mean(weights)

    @staticmethod
    def logit_mean_probability(outputs):
        raw_probs = [o["confidences"]["raw_confidences"] for o in outputs]
        mean_prob = np.exp(np.mean(np.log(raw_probs))) if raw_probs else 0
        return AnswerPostprocessor.majority_vote(outputs)[0], mean_prob

    @staticmethod
    def bayesian_aggregation(outputs):
        # Placeholder for Bayesian aggregation logic
        return AnswerPostprocessor.majority_vote(outputs)

    def __init__(self, folder_path="."):
        self.folder_path = folder_path

    def process_all(self):
        input_files = glob.glob(f"{self.folder_path}/answers_*.jsonl")
        for input_file in input_files:
            # Implement file reading, cleaning, aggregation, and saving logic as in original script
            pass  # Placeholder for full file processing logic

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Clean, aggregate, and postprocess model answer outputs in .jsonl files.")
    parser.add_argument('--folder_path', type=str, default=".", help='Folder containing answers_*.jsonl files')
    args = parser.parse_args()
    postprocessor = AnswerPostprocessor(args.folder_path)
    postprocessor.process_all()
