from ..data.cleaner import DataCleaner

class OutputCleaner:
    """
    A professional, extensible class for cleaning model outputs according to question type.
    Delegates to DataCleaner for type-specific cleaning. Provides detailed documentation and explicit logic.
    """
    @staticmethod
    def clean_output(value: str, q_type: str) -> str:
        """
        Clean a model output string according to the question type.
        Args:
            value (str): The output string to clean.
            q_type (str): The type of question ('binary', 'numeric', 'date', etc.).
        Returns:
            str: Cleaned output string.
        """
        if q_type == "binary":
            return DataCleaner.clean_binary(value)
        elif q_type == "numeric":
            return DataCleaner.clean_numeric(value)
        elif q_type == "date":
            return DataCleaner.clean_date(value)
        else:
            # If question type is unrecognized, return the value as-is
            return value
