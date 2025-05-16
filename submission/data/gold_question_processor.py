import os
import json
import math

class GoldQuestionProcessor:
    """
    A professional, extensible class for processing raw gold question data. Extracts question ID, title, resolution, timestamps,
    and computes a normalized score using a sigmoid transformation. Designed for clarity, maintainability, and extensibility.
    """

    @staticmethod
    def sigmoid(x: float) -> float:
        """
        Compute the sigmoid function for a given value.
        Args:
            x (float): Input value.
        Returns:
            float: The sigmoid of x.
        """
        return 1 / (1 + math.exp(-x))

    def __init__(self, input_dir: str = "filtered_questions", output_file: str = "gold.jsonl"):
        """
        Initialize the GoldQuestionProcessor.
        Args:
            input_dir (str): Directory containing filtered question JSON files (by category).
            output_file (str): Output JSONL file to store processed data.
        """
        self.input_dir = input_dir
        self.output_file = output_file
        self.categories = ["binary", "numeric", "date"]

    def process_raw_data(self) -> None:
        """
        Process raw gold question data for each category, extract relevant fields, normalize scores,
        and write the results to the output file.
        """
        processed_data = []
        for category in self.categories:
            # Construct the path to the filtered questions file for this category
            input_path = os.path.join(self.input_dir, category, f"{category}_questions_filtered.json")
            if not os.path.exists(input_path):
                print(f"Warning: {input_path} not found, skipping.")
                continue
            with open(input_path, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    for entry in data:
                        question_data = entry.get("question", {})
                        # Extract the baseline score from nested dictionaries, if present
                        baseline_score = (
                            question_data.get("aggregations", {})
                            .get("recency_weighted", {})
                            .get("score_data", {})
                            .get("baseline_score")
                        )
                        # Normalize the score using the sigmoid function, if available
                        if baseline_score is not None:
                            normalized_score = self.sigmoid(baseline_score / 100)
                        else:
                            normalized_score = None
                        # Collect all relevant fields into a dictionary
                        processed_data.append({
                            "id": question_data.get("id"),
                            "question": question_data.get("title"),
                            "type": category,
                            "resolution": question_data.get("resolution"),
                            "resolution_time": question_data.get("actual_resolve_time"),
                            "created_time": question_data.get("created_at"),
                            "normalized_score": normalized_score
                        })
                except json.JSONDecodeError:
                    print(f"Error decoding JSON in {input_path}")
                    continue
        # Write the processed data to the output file, one JSON object per line
        with open(self.output_file, "w", encoding="utf-8") as f:
            for item in processed_data:
                f.write(json.dumps(item) + "\n")
        print(f"✅ Processed data saved to {self.output_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process raw gold question data and extract id, title, resolution, times, and normalized score.")
    parser.add_argument('--input_dir', type=str, default="filtered_questions", help='Input directory containing filtered question JSON files')
    parser.add_argument('--output_file', type=str, default="gold.jsonl", help='Output JSONL file to store processed data')
    args = parser.parse_args()
    processor = GoldQuestionProcessor(args.input_dir, args.output_file)
    processor.process_raw_data()
