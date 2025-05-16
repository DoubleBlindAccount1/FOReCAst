import pandas as pd

class CutoffMatchResultsFilter:
    """
    A professional, extensible class for filtering and producing summary tables for RQ1 based on model cutoff dates and aggregation method, requiring filter_date to exactly match the cutoff date for each model.
    This class loads CSV results for binary, numeric, and date question types, and provides methods to filter and summarize
    results for publication or reporting. All logic is thoroughly documented for clarity and future maintainability.
    """
    # Mapping from model short names to their cutoff dates for filtering
    cutoff_dates = {
        "gpt2": "2017-12-01",
        "gpt2-xl": "2017-12-01",
        "pythia-2.8b": "2020-03-01",
        "pythia-14m": "2020-03-01",
        "pythia-160m": "2020-03-01",
        "bloom-7b1": "2021-12-01",
        "bloom-560m": "2021-12-01",
        "llama-7b": "2022-08-01",
        "OLMo-7B": "2023-03-01",
        "OLMo-7B-Instruct-hf": "2023-03-01",
        "OLMo-1B": "2023-03-01",
        "CodeLlama-7b-Instruct-hf": "2023-03-01",
        "OLMo-2-1124-7B": "2023-12-01",
        "OLMo-2-1124-7B-Instruct": "2023-12-01"
    }

    @classmethod
    def get_short_model(cls, model_str: str) -> str:
        """
        Extract the short model name from a full model string using known cutoff date keys.
        Args:
            model_str (str): The full model string.
        Returns:
            str: The short model name if found, otherwise None.
        """
        for key in cls.cutoff_dates:
            if model_str.endswith(key):
                return key
        return None

    @classmethod
    def filter_by_cutoff(cls, df: pd.DataFrame, expected_question_type: str) -> pd.DataFrame:
        """
        Filter a DataFrame for rows matching the specified question type, filter type, and aggregation result.
        Only include rows where the filter_date exactly matches the cutoff date for the model.
        Excludes models containing 'CodeLlama' in their name (case-insensitive).
        Args:
            df (pd.DataFrame): The DataFrame to filter.
            expected_question_type (str): The question type to filter for ('binary', 'numeric', or 'date').
        Returns:
            pd.DataFrame: The filtered DataFrame.
        """
        df_filtered = df[
            (df['filter_type'] == 'resolution_time') &
            (df['agg_result'] == 'highest_confidence') &
            (df['question_type'] == expected_question_type)
        ]
        # Exclude models containing 'CodeLlama' (case-insensitive)
        df_filtered = df_filtered[~df_filtered['model'].str.contains("CodeLlama", case=False, na=False)]
        def row_matches_cutoff(row):
            short_model = cls.get_short_model(row['model'])
            if short_model is None:
                return False
            expected_date = cls.cutoff_dates.get(short_model)
            return row['filter_date'] == expected_date
        df_filtered = df_filtered[df_filtered.apply(row_matches_cutoff, axis=1)]
        return df_filtered

    def __init__(self, binary_csv_path: str, numeric_csv_path: str, date_csv_path: str):
        """
        Initialize the CutoffMatchResultsFilter by loading the required CSV files for each question type.
        Args:
            binary_csv_path (str): Path to the CSV file containing binary question results.
            numeric_csv_path (str): Path to the CSV file containing numeric question results.
            date_csv_path (str): Path to the CSV file containing date question results.
        """
        self.df_binary = pd.read_csv(binary_csv_path)
        self.df_numeric = pd.read_csv(numeric_csv_path)
        self.df_date = pd.read_csv(date_csv_path)

    def get_tables(self):
        """
        Generate summary tables for RQ1 for binary, numeric, and date question types, filtered by model cutoff criteria.
        Returns:
            tuple: (binary_table, numeric_table, date_table) as pandas DataFrames.
        """
        df_binary_final = self.filter_by_cutoff(self.df_binary, 'binary')
        df_numeric_final = self.filter_by_cutoff(self.df_numeric, 'numeric')
        df_date_final = self.filter_by_cutoff(self.df_date, 'date')
        # Define the columns to include in each summary table
        binary_columns = ['model', 'filter_date', 'num_questions', 'accuracy', 'f1', 'brier']
        numeric_columns = ['model', 'filter_date', 'num_questions', 'norm_ape', 'norm_mae', 'norm_crps']
        date_columns = ['model', 'filter_date', 'num_questions', 'norm_days_error', 'norm_crps']
        # Prepare and round each table for output
        binary_table = df_binary_final[binary_columns].copy().round(4)
        numeric_table = df_numeric_final[numeric_columns].copy().round(4)
        date_table = df_date_final[date_columns].copy().round(4)
        # Sort tables by filter_date for clarity
        binary_table.sort_values(by='filter_date', inplace=True)
        numeric_table.sort_values(by='filter_date', inplace=True)
        date_table.sort_values(by='filter_date', inplace=True)
        return binary_table, numeric_table, date_table

    def save_tables(self, binary_out_path: str, numeric_out_path: str, date_out_path: str) -> None:
        """
        Save the RQ1 summary tables to CSV files for binary, numeric, and date question types.
        Args:
            binary_out_path (str): Output path for the binary questions table.
            numeric_out_path (str): Output path for the numeric questions table.
            date_out_path (str): Output path for the date questions table.
        """
        binary_table, numeric_table, date_table = self.get_tables()
        binary_table.to_csv(binary_out_path, index=False)
        numeric_table.to_csv(numeric_out_path, index=False)
        date_table.to_csv(date_out_path, index=False)

    def print_tables(self) -> None:
        """
        Print the RQ1 summary tables for binary, numeric, and date question types to the console.
        """
        binary_table, numeric_table, date_table = self.get_tables()
        print("=== RQ1: Binary Questions Table (Cutoff Match) ===")
        print(binary_table.to_string(index=False))
        print("\n=== RQ1: Numeric Questions Table (Cutoff Match) ===")
        print(numeric_table.to_string(index=False))
        print("\n=== RQ1: Date Questions Table (Cutoff Match) ===")
        print(date_table.to_string(index=False))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Filter and summarize results for RQ1 based on model cutoff dates and aggregation method, requiring filter_date to match cutoff.")
    parser.add_argument('--binary_csv', type=str, default="overall_results_binary.csv")
    parser.add_argument('--numeric_csv', type=str, default="overall_results_numeric.csv")
    parser.add_argument('--date_csv', type=str, default="overall_results_date.csv")
    parser.add_argument('--binary_out', type=str, default="RQ1_binary_cutoff_match_table.csv")
    parser.add_argument('--numeric_out', type=str, default="RQ1_numeric_cutoff_match_table.csv")
    parser.add_argument('--date_out', type=str, default="RQ1_date_cutoff_match_table.csv")
    args = parser.parse_args()
    filterer = CutoffMatchResultsFilter(args.binary_csv, args.numeric_csv, args.date_csv)
    filterer.save_tables(args.binary_out, args.numeric_out, args.date_out)
    filterer.print_tables()
