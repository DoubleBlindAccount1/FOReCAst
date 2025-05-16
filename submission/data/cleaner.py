import unicodedata
import re
from word2number import w2n
from typing import Any

class DataCleaner:
    """
    A professional, extensible class for cleaning and normalizing text, binary, numeric, and date data fields.
    All methods are static for easy reuse. Extensive comments and explicit logic are provided for clarity and maintainability.
    """

    @staticmethod
    def clean_text(text: Any) -> str:
        """
        Normalize and clean a text string by removing special Unicode characters, normalizing quotes,
        and trimming whitespace. If the input is not a string, returns it unchanged.
        Args:
            text (Any): The text to clean.
        Returns:
            str: The cleaned text string.
        """
        if not isinstance(text, str):
            # If input is not a string, return as-is
            return text
        # Normalize Unicode to NFKC form
        text = unicodedata.normalize("NFKC", text)
        # Replace curly quotes with straight quotes
        text = text.replace("\u201c", '"').replace("\u201d", '"').replace("\u2018", "'").replace("\u2019", "'")
        # Remove zero-width and invisible Unicode characters
        text = re.sub(r"[\u200B-\u200D\uFEFF]", "", text)
        # Trim whitespace
        return text.strip()

    @staticmethod
    def clean_binary(value: str) -> str:
        """
        Clean and normalize binary (yes/no) answers.
        Args:
            value (str): The answer string to clean.
        Returns:
            str: Canonical binary answer ('yes', 'no', or 'unclear').
        """
        value = value.lower().strip()
        # Mapping of various affirmative/negative expressions to canonical binary answers
        binary_map = {
            "yes": "yes", "no": "no", "true": "yes", "false": "no",
            "sure": "yes", "absolutely": "yes", "certainly": "yes", "correct": "yes",
            "of course": "yes", "indeed": "yes", "affirmative": "yes",
            "not at all": "no", "unlikely": "no", "no way": "no",
            "never": "no", "impossible": "no", "incorrect": "no"
        }
        for word, mapped in binary_map.items():
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
            except Exception:
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
