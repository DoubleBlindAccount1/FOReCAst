import pandas as pd
import argparse

class LatexTableGenerator:
    """
    A utility class for generating LaTeX tables from CSV files containing model results.
    Provides methods for cleaning model names, rounding numeric values, and formatting
    DataFrames as LaTeX tables suitable for publication. All methods are static as no
    instance state is required.
    """

    @staticmethod
    def clean_model_names(dataframe):
        """
        Clean the 'model' column in the DataFrame by removing any prefix before the last underscore.
        For example, converts 'allenai_OLMo-7B-Instruct-hf' to 'OLMo-7B-Instruct-hf'.
        This is useful for making model names more readable in output tables.

        Args:
            dataframe (pd.DataFrame): The DataFrame to process. Must contain a 'model' column.

        Returns:
            pd.DataFrame: The DataFrame with cleaned 'model' names.
        """
        if 'model' in dataframe.columns:
            # Iterate over each value in the 'model' column and clean the name
            cleaned_model_names = []
            for model_name in dataframe['model']:
                # Split the model name by '_' and take the last part for clarity
                if isinstance(model_name, str):
                    split_parts = model_name.split('_')
                    cleaned_name = split_parts[-1]
                    cleaned_model_names.append(cleaned_name)
                else:
                    cleaned_model_names.append(model_name)
            dataframe['model'] = cleaned_model_names
        return dataframe

    @staticmethod
    def round_numeric_values(dataframe):
        """
        Round all numeric columns in the DataFrame to 4 decimal places for consistency in reporting.
        Non-numeric columns are left unchanged. The rounding is performed as string formatting
        to ensure LaTeX tables display values with a fixed number of decimals.

        Args:
            dataframe (pd.DataFrame): The DataFrame to process.

        Returns:
            pd.DataFrame: The DataFrame with rounded numeric values as strings.
        """
        for column_name in dataframe.columns:
            # Check if the column contains numeric data
            if pd.api.types.is_numeric_dtype(dataframe[column_name]):
                # Create a new list to hold rounded values for this column
                rounded_values = []
                for value in dataframe[column_name]:
                    try:
                        # Format the value to 4 decimal places
                        rounded_value = f"{float(value):.4f}"
                    except (ValueError, TypeError):
                        # If conversion fails, keep the original value
                        rounded_value = value
                    rounded_values.append(rounded_value)
                dataframe[column_name] = rounded_values
        return dataframe

    @staticmethod
    def to_latex_table(dataframe: pd.DataFrame, caption: str, label: str) -> str:
        """
        Convert a DataFrame into a LaTeX table string, including formatting for publication.
        The table uses the 'booktabs' package conventions, includes a caption and label,
        and centers content. All columns are center-aligned.

        Args:
            dataframe (pd.DataFrame): The DataFrame to convert to LaTeX.
            caption (str): The caption for the LaTeX table.
            label (str): The LaTeX label for referencing the table.

        Returns:
            str: The LaTeX code for the table.
        """
        # Determine the number of columns and set a centered format for each column
        number_of_columns = len(dataframe.columns)
        column_format = "c" * number_of_columns
        # Start building the LaTeX table string
        latex_lines = []
        latex_lines.append("\\begin{table}[H]")
        latex_lines.append("\\centering")
        latex_lines.append("\\footnotesize")
        latex_lines.append(f"\\begin{{tabular}}{{{column_format}}}")
        latex_lines.append("\\toprule")
        # Header row
        header_row = " & ".join([str(col) for col in dataframe.columns]) + " \\\\"  # LaTeX row ending
        latex_lines.append(header_row)
        latex_lines.append("\\midrule")
        # Data rows
        for row_index in range(len(dataframe)):
            row = dataframe.iloc[row_index]
            row_values = []
            for value in row:
                row_values.append(str(value))
            row_string = " & ".join(row_values) + " \\\\"  # LaTeX row ending
            latex_lines.append(row_string)
        latex_lines.append("\\bottomrule")
        latex_lines.append("\\end{tabular}")
        latex_lines.append(f"\\caption{{{caption}}}")
        latex_lines.append(f"\\label{{{label}}}")
        latex_lines.append("\\end{table}")
        # Join all lines with newline characters
        latex_code = "\n".join(latex_lines)
        return latex_code

    @staticmethod
    def csv_to_latex_table(csv_filename, caption, label):
        """
        Read a CSV file, clean the model names, round numeric values to 4 decimals,
        and convert the DataFrame into LaTeX table code. This is a high-level utility
        for quickly generating publication-ready tables from result CSVs.

        Args:
            csv_filename (str): Path to the input CSV file.
            caption (str): Caption for the LaTeX table.
            label (str): Label for referencing the table in LaTeX.

        Returns:
            str: The LaTeX table code as a string.
        """
        # Step 1: Read the CSV file into a DataFrame
        dataframe = pd.read_csv(csv_filename)

        # Step 2: Clean the model names to remove prefixes for clarity
        dataframe = LatexTableGenerator.clean_model_names(dataframe)

        # Step 3: Round all numeric columns to 4 decimal places for consistency
        dataframe = LatexTableGenerator.round_numeric_values(dataframe)

        # Step 4: Convert the DataFrame to LaTeX table code
        latex_code = LatexTableGenerator.to_latex_table(dataframe, caption, label)

        return latex_code

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate LaTeX tables from CSV results.")
    parser.add_argument('--binary_csv', type=str, default='overall_results_binary.csv', help='CSV file for binary results')
    parser.add_argument('--numeric_csv', type=str, default='overall_results_numeric.csv', help='CSV file for numeric results')
    parser.add_argument('--date_csv', type=str, default='overall_results_date.csv', help='CSV file for date results')
    parser.add_argument('--out_binary', type=str, default='binary_table.txt', help='Output file for binary LaTeX table')
    parser.add_argument('--out_numeric', type=str, default='numeric_table.txt', help='Output file for numeric LaTeX table')
    parser.add_argument('--out_date', type=str, default='date_table.txt', help='Output file for date LaTeX table')
    args = parser.parse_args()

    # Generate LaTeX table code for each CSV file.
    binary_table_latex = LatexTableGenerator.csv_to_latex_table(
        args.binary_csv, "Overall Results (Binary Questions)", "tab:overall_results_binary")
    numeric_table_latex = LatexTableGenerator.csv_to_latex_table(
        args.numeric_csv, "Overall Results (Numeric Questions)", "tab:overall_results_numeric")
    date_table_latex = LatexTableGenerator.csv_to_latex_table(
        args.date_csv, "Overall Results (Date Questions)", "tab:overall_results_date")

    # Save each LaTeX code to a text file because the tables may be large.
    with open(args.out_binary, "w") as f:
        f.write(binary_table_latex)
    with open(args.out_numeric, "w") as f:
        f.write(numeric_table_latex)
    with open(args.out_date, "w") as f:
        f.write(date_table_latex)

    print(f"LaTeX code for each table has been saved to {args.out_binary}, {args.out_numeric}, and {args.out_date}.")
