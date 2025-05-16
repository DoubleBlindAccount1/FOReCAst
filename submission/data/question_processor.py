import os
import json

class QuestionProcessor:
    """
    A professional, extensible class for processing raw question data. Extracts question ID, title, and type for each question.
    Designed for clarity, maintainability, and extensibility. Equivalent to the logic in process_data.py, but as a reusable class.
    """

    def __init__(self, input_dir: str = "filtered_questions", output_file: str = "questions.jsonl"):
        """
        Initialize the QuestionProcessor.
        Args:
            input_dir (str): Directory containing filtered question JSON files (by category).
            output_file (str): Output JSONL file to store processed data.
        """
        self.input_dir = input_dir
        self.output_file = output_file
        self.categories = ["binary", "numeric", "date"]

    def process_raw_data(self) -> None:
        """
        Process raw question data for each category, extract relevant fields (ID, title, type),
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
                        # Extract the required fields from each question entry
                        processed_data.append({
                            "id": entry["question"]["id"],
                            "question": entry["question"]["title"],
                            "type": category
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
    parser = argparse.ArgumentParser(description="Process raw question data and extract id, title, and description for each question type.")
    parser.add_argument('--input_dir', type=str, default="filtered_questions", help='Input directory containing filtered question JSON files')
    parser.add_argument('--output_file', type=str, default="questions.jsonl", help='Output JSONL file to store processed data')
    args = parser.parse_args()
    processor = QuestionProcessor(args.input_dir, args.output_file)
    processor.process_raw_data()
