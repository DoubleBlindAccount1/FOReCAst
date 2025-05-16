import pandas as pd

import pandas as pd

class AggregationResultsFilter:
    """
    A professional, extensible class for filtering and producing summary tables for RQ5 based on aggregation methods.
    This class loads CSV results for binary, numeric, and date question types, and provides methods to filter and summarize
    results for publication or reporting. All logic is thoroughly documented for clarity and future maintainability.
    """
    def __init__(self, binary_csv_path: str, numeric_csv_path: str, date_csv_path: str):
        """
        Initialize the AggregationResultsFilter by loading the required CSV files for each question type.
        Args:
            binary_csv_path (str): Path to the CSV file containing binary question results.
            numeric_csv_path (str): Path to the CSV file containing numeric question results.
            date_csv_path (str): Path to the CSV file containing date question results.
        """
        self.df_binary = pd.read_csv(binary_csv_path)
        self.df_numeric = pd.read_csv(numeric_csv_path)
        self.df_date = pd.read_csv(date_csv_path)

    @staticmethod
    def filter_by_aggregation(
        dataframe: pd.DataFrame,
        question_type: str,
        expected_filter_date: str = "2022-08-01",
        model_substr: str = None
    ) -> pd.DataFrame:
        """
        Filter the given DataFrame for records matching the specified question type, filter type, and date.
        Optionally filter by a model name substring (case-insensitive).
        Args:
            dataframe (pd.DataFrame): The DataFrame to filter.
            question_type (str): The type of question ('binary', 'numeric', or 'date').
            expected_filter_date (str): The expected filter date (default: '2022-08-01').
            model_substr (str, optional): Substring to filter model names (case-insensitive).
        Returns:
            pd.DataFrame: The filtered DataFrame.
        """
        # Build the mask for filter_type, question_type, and filter_date
        mask = (
            (dataframe['filter_type'] == 'resolution_time') &
            (dataframe['question_type'] == question_type) &
            (dataframe['filter_date'] == expected_filter_date)
        )
        # Optionally filter by model substring
        if model_substr is not None:
            mask = mask & (dataframe['model'].str.contains(model_substr, case=False, na=False))
        return dataframe[mask]

    def get_rq5_tables(
        self,
        model_substr: str = None,
        expected_filter_date: str = "2022-08-01"
    ):
        """
        Generate summary tables for RQ5 for binary, numeric, and date question types, optionally filtered by model substring and date.
        Args:
            model_substr (str, optional): Substring to filter model names.
            expected_filter_date (str): The filter date to use (default: '2022-08-01').
        Returns:
            tuple: (binary_table, numeric_table, date_table) as pandas DataFrames.
        """
        # Filter each DataFrame for RQ5
        df_binary_rq5 = self.filter_by_aggregation(self.df_binary, "binary", expected_filter_date, model_substr)
        df_numeric_rq5 = self.filter_by_aggregation(self.df_numeric, "numeric", expected_filter_date, model_substr)
        df_date_rq5 = self.filter_by_aggregation(self.df_date, "date", expected_filter_date, model_substr)
        # Define the columns to include in each summary table
        binary_columns = ['model', 'agg_result', 'num_questions', 'accuracy', 'f1', 'brier']
        numeric_columns = ['model', 'agg_result', 'num_questions', 'norm_ape', 'norm_mae', 'norm_crps']
        date_columns = ['model', 'agg_result', 'num_questions', 'norm_crps', 'norm_days_error']
        # Prepare and round each table for output
        table_binary = df_binary_rq5[binary_columns].copy().round(4)
        table_numeric = df_numeric_rq5[numeric_columns].copy().round(4)
        table_date = df_date_rq5[date_columns].copy().round(4)
        return table_binary, table_numeric, table_date

    def save_tables(
        self,
        binary_out_path: str,
        numeric_out_path: str,
        date_out_path: str,
        model_substr: str = None,
        expected_filter_date: str = "2022-08-01"
    ) -> None:
        """
        Save the RQ5 summary tables to CSV files for binary, numeric, and date question types.
        Args:
            binary_out_path (str): Output path for the binary questions table.
            numeric_out_path (str): Output path for the numeric questions table.
            date_out_path (str): Output path for the date questions table.
            model_substr (str, optional): Substring to filter model names.
            expected_filter_date (str): The filter date to use.
        """
        table_binary, table_numeric, table_date = self.get_rq5_tables(model_substr, expected_filter_date)
        table_binary.to_csv(binary_out_path, index=False)
        table_numeric.to_csv(numeric_out_path, index=False)
        table_date.to_csv(date_out_path, index=False)

    def print_tables(
        self,
        model_substr: str = None,
        expected_filter_date: str = "2022-08-01"
    ) -> None:
        """
        Print the RQ5 summary tables for binary, numeric, and date question types to the console.
        Args:
            model_substr (str, optional): Substring to filter model names.
            expected_filter_date (str): The filter date to use.
        """
        table_binary, table_numeric, table_date = self.get_rq5_tables(model_substr, expected_filter_date)
        print("=== RQ5: Binary Questions Table ===")
        print(table_binary.to_string(index=False))
        print("\n=== RQ5: Numeric Questions Table ===")
        print(table_numeric.to_string(index=False))
        print("\n=== RQ5: Date Questions Table ===")
        print(table_date.to_string(index=False))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Filter and summarize results for RQ5 based on aggregation methods.")
    parser.add_argument('--binary_csv', type=str, default="overall_results_binary.csv")
    parser.add_argument('--numeric_csv', type=str, default="overall_results_numeric.csv")
    parser.add_argument('--date_csv', type=str, default="overall_results_date.csv")
    parser.add_argument('--binary_out', type=str, default="RQ5_binary_table.csv")
    parser.add_argument('--numeric_out', type=str, default="RQ5_numeric_table.csv")
    parser.add_argument('--date_out', type=str, default="RQ5_date_table.csv")
    parser.add_argument('--model_substr', type=str, default=None, help='Optional substring to filter models (e.g. "llama-7b")')
    parser.add_argument('--filter_date', type=str, default="2022-08-01", help='Expected filter date (YYYY-MM-DD)')
    args = parser.parse_args()
    filterer = AggregationResultsFilter(args.binary_csv, args.numeric_csv, args.date_csv)
    filterer.save_tables(args.binary_out, args.numeric_out, args.date_out, args.model_substr, args.filter_date)
    filterer.print_tables(args.model_substr, args.filter_date)
