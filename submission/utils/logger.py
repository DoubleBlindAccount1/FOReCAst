import datetime

class Logger:
    """
    A professional, extensible logging utility class for standardized console output.
    Provides static methods for info, warning, and error messages, each with a timestamp.
    Designed for clarity and maintainability.
    """
    @staticmethod
    def info(msg: str):
        """
        Print an informational message with a timestamp.
        Args:
            msg (str): The message to print.
        """
        print(f"[INFO {Logger._now()}] {msg}")

    @staticmethod
    def warning(msg: str):
        """
        Print a warning message with a timestamp.
        Args:
            msg (str): The message to print.
        """
        print(f"[WARNING {Logger._now()}] {msg}")

    @staticmethod
    def error(msg: str):
        """
        Print an error message with a timestamp.
        Args:
            msg (str): The message to print.
        """
        print(f"[ERROR {Logger._now()}] {msg}")

    @staticmethod
    def _now() -> str:
        """
        Get the current timestamp as an ISO-formatted string (up to seconds).
        Returns:
            str: Current timestamp in ISO format.
        """
        return datetime.datetime.now().isoformat(timespec='seconds')
