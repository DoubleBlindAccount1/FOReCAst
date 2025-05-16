import os
import json
from collections import Counter, defaultdict
from datetime import datetime

class QuestionStatsAnalyzer:
    """
    A professional, extensible analyzer for collecting and printing statistics from filtered question JSON files.
    This class computes month and category distributions both globally and per file, and is thoroughly documented for clarity and future maintainability.
    """
    def __init__(self, base_folder: str = "filtered_questions"):
        """
        Initialize the QuestionStatsAnalyzer.
        Args:
            base_folder (str): The root directory containing filtered question JSON files.
        """
        self.base_folder = base_folder
        self.global_months = []  # List of all months from all files
        self.global_categories = []  # List of all categories from all files
        self.per_file_data = {}  # Mapping: filename -> (month list, category list)

    @staticmethod
    def extract_data_from_file(filepath: str):
        """
        Extract months and categories from a single JSON file.
        Args:
            filepath (str): Path to the JSON file.
        Returns:
            tuple: (list of months, list of categories)
        """
        with open(filepath, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: Failed to parse {filepath}")
                return [], []
        months = []
        categories = []
        for entry in data:
            # Extract month from actual_close_time
            time_str = entry.get("actual_close_time")
            if time_str:
                try:
                    date_obj = datetime.fromisoformat(time_str.replace("Z", "+00:00"))
                    month_str = date_obj.strftime("%Y-%m")
                    months.append(month_str)
                except ValueError:
                    # Skip if the date format is invalid
                    continue
            # Extract categories from nested structure
            cat_list = entry.get("projects", {}).get("category", [])
            for cat in cat_list:
                if isinstance(cat, dict) and "name" in cat:
                    categories.append(cat["name"])
        return months, categories

    def collect_all_data(self) -> None:
        """
        Walk through all JSON files in the base folder, extract months and categories,
        and aggregate them globally and per file.
        """
        self.global_months = []
        self.global_categories = []
        self.per_file_data = {}
        for root, _, files in os.walk(self.base_folder):
            for filename in files:
                if filename.endswith(".json"):
                    filepath = os.path.join(root, filename)
                    months, categories = self.extract_data_from_file(filepath)
                    self.global_months.extend(months)
                    self.global_categories.extend(categories)
                    self.per_file_data[filepath] = (months, categories)

    @staticmethod
    def print_top_distribution(counter: Counter, top_n: int = 10) -> None:
        """
        Print the top N items from a Counter, with counts and percentages.
        Args:
            counter (Counter): Counter object with item frequencies.
            top_n (int): Number of top items to print.
        """
        total = sum(counter.values())
        if total == 0:
            print("  (No data)")
            return
        for item, count in counter.most_common(top_n):
            percent = 100 * count / total
            print(f"  {item}: {count} ({percent:.2f}%)")

    def print_stats(self) -> None:
        """
        Print global and per-file statistics for months and categories.
        """
        print("=== Global Distribution ===")
        print("Top 10 Months (by actual_close_time):")
        self.print_top_distribution(Counter(self.global_months))
        print("\nTop 10 Categories:")
        self.print_top_distribution(Counter(self.global_categories))
        print("\n=== Per-File Distribution ===")
        for filepath, (months, categories) in self.per_file_data.items():
            print(f"\nFile: {filepath}")
            print("  Top Months:")
            self.print_top_distribution(Counter(months))
            print("  Top Categories:")
            self.print_top_distribution(Counter(categories))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Collect and print statistics from filtered question JSON files.")
    parser.add_argument('--base_folder', type=str, default="filtered_questions", help='Folder containing filtered question JSON files')
    args = parser.parse_args()
    analyzer = QuestionStatsAnalyzer(args.base_folder)
    analyzer.collect_all_data()
    analyzer.print_stats()
