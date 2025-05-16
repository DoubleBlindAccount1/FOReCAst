from submission.data.loader import JsonlDataLoader
from submission.data.cleaner import DataCleaner
from submission.data.splitter import DataSplitter
from submission.data.gold_processor import GoldDataProcessor
from submission.models.manager import ModelManager
from submission.models.inference import InferenceEngine
from submission.evaluation.metrics import AccuracyMetric, F1ScoreMetric
from submission.evaluation.aggregator import MajorityVoteAggregator
from submission.evaluation.filters import CutoffDateFilter
from submission.evaluation.evaluator import PredictionEvaluator
from submission.postprocessing.parser import OutputParser
from submission.postprocessing.cleaner import OutputCleaner
from submission.postprocessing.latex_table import LatexTableGenerator
from submission.utils.logger import Logger
from submission.utils.config import Config

import pandas as pd
import argparse
import sys

def main():
    """
    Main entry point for the FOReCAst OOP pipeline. Handles argument parsing, data loading, cleaning, splitting,
    model inference, postprocessing, evaluation, and LaTeX table generation.
    Designed for clarity, extensibility, and maintainability.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="FOReCAst OOP Pipeline")
    parser.add_argument('--train', type=str, default=Config.TRAIN_FILE, help='Path to train file (jsonl)')
    parser.add_argument('--dev', type=str, default=Config.DEV_FILE, help='Path to dev file (jsonl)')
    parser.add_argument('--test', type=str, default=Config.TEST_FILE, help='Path to test file (jsonl)')
    parser.add_argument('--gold', type=str, default=Config.GOLD_FILE, help='Path to gold file (jsonl)')
    parser.add_argument('--model', type=str, default='meta-llama/Llama-3.1-8B', help='Model name for inference')
    parser.add_argument('--train_ratio', type=float, default=0.65, help='Train split ratio')
    parser.add_argument('--dev_ratio', type=float, default=0.1, help='Dev split ratio')
    parser.add_argument('--test_ratio', type=float, default=0.25, help='Test split ratio')
    parser.add_argument('--output_table', type=str, default='results_table.tex', help='Path to output LaTeX table')
    parser.add_argument('--caption', type=str, default='Results Table', help='LaTeX table caption')
    parser.add_argument('--label', type=str, default='tab:results', help='LaTeX table label')
    args = parser.parse_args()

    Logger.info("Starting FOReCAst OOP pipeline (CLI mode)...")

    # 1. Load and clean data
    Logger.info(f"Loading training data from {args.train}...")
    loader = JsonlDataLoader()
    raw_data = loader.load(args.train)
    Logger.info(f"Loaded {len(raw_data)} records. Cleaning questions...")
    cleaned_data = []
    for entry in raw_data:
        # Clean the question text using DataCleaner
        cleaned_entry = dict(entry)
        cleaned_entry["question"] = DataCleaner.clean_text(entry["question"])
        cleaned_data.append(cleaned_entry)
    Logger.info(f"Cleaned {len(cleaned_data)} records.")

    # 2. Split data
    Logger.info("Splitting data into train/dev/test sets...")
    train, dev, test = DataSplitter.split_dataset(
        cleaned_data,
        train_ratio=args.train_ratio,
        dev_ratio=args.dev_ratio,
        test_ratio=args.test_ratio
    )
    Logger.info(f"Train: {len(train)}, Dev: {len(dev)}, Test: {len(test)}")

    # 3. Process gold data
    Logger.info("Processing gold data...")
    gold_data = GoldDataProcessor.process_gold_data(raw_data)

    # 4. Download/load model
    Logger.info(f"Downloading/loading model: {args.model}")
    model_manager = ModelManager()
    model_name = args.model
    model_manager.download_model(model_name)
    model, tokenizer = model_manager.load_model_and_tokenizer(model_name)

    # 5. Inference
    Logger.info("Running inference on test set...")
    engine = InferenceEngine(model, tokenizer)
    predictions = engine.infer(test)
    Logger.info(f"Inference complete. Generated {len(predictions)} predictions.")

    # 6. Postprocess outputs
    Logger.info("Postprocessing outputs...")
    parsed = [OutputParser.extract_value_confidence(q["type"], q["question"], q["generated"]) for q in predictions]
    cleaned_outputs = [OutputCleaner.clean_output(p["value"], q["type"]) for p, q in zip(parsed, predictions)]

    # 7. Evaluation
    Logger.info("Evaluating predictions...")
    metrics = [AccuracyMetric(), F1ScoreMetric()]
    aggregator = MajorityVoteAggregator()
    cutoff_dates = {"llama-7b": "2022-08-01"}  # Example cutoff date mapping
    filters = [CutoffDateFilter(cutoff_dates)]
    evaluator = PredictionEvaluator(metrics, aggregator, filters)
    eval_results = evaluator.evaluate(predictions, gold_data)
    Logger.info(f"Evaluation Results: {eval_results}")

    # 8. Generate LaTeX table
    Logger.info("Generating LaTeX table...")
    df = pd.DataFrame([p for p in predictions])
    latex = LatexTableGenerator.to_latex_table(df, caption=args.caption, label=args.label)
    with open(args.output_table, 'w', encoding='utf-8') as f:
        f.write(latex)
    Logger.info(f"LaTeX Table saved to {args.output_table}")

if __name__ == "__main__":
    main()
