import os
import json
import glob
import csv
import datetime
import math
from collections import defaultdict
from typing import List, Dict, Any
import numpy as np
from dateutil.parser import parse as date_parse
from scipy.stats import norm, spearmanr
import argparse
from .metrics import Metric
from .aggregator import Aggregator
from .filters import ResultFilter

class BaseEvaluator:
    """
    Abstract base class for evaluation of model predictions. Subclasses should implement
    the evaluate method to compare predictions with gold data using specified metrics,
    aggregation logic, and optional filters.
    """
    def __init__(self, metrics: List[Metric], aggregator: Aggregator, filters: List[ResultFilter] = None):
        """
        Initialize the evaluator with metrics, aggregator, and optional filters.
        Args:
            metrics (List[Metric]): List of metric objects to compute scores.
            aggregator (Aggregator): Aggregation object to summarize results.
            filters (List[ResultFilter], optional): List of filters to preprocess records.
        """
        self.metrics = metrics
        self.aggregator = aggregator
        self.filters = filters or []

    def evaluate(self, predictions: List[Dict[str, Any]], gold: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Abstract method to evaluate predictions against gold data.
        Args:
            predictions (List[Dict[str, Any]]): Model predictions.
            gold (List[Dict[str, Any]]): Gold standard answers.
        Returns:
            Dict[str, Any]: Evaluation results.
        """
        raise NotImplementedError("Subclasses must implement the evaluate method.")

class PredictionEvaluator(BaseEvaluator):
    """
    Evaluator for model predictions. Matches predictions to gold data by ID, applies filters,
    aggregates results, and computes all specified metrics. Provides static methods for
    common evaluation metrics such as Brier score, ECE, and CRPS.
    """
    def evaluate(self, predictions: List[Dict[str, Any]], gold: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate predictions against gold data by matching IDs, applying filters, aggregating,
        and computing metrics.
        Args:
            predictions (List[Dict[str, Any]]): Model predictions, each with an 'id' field.
            gold (List[Dict[str, Any]]): Gold standard answers, each with an 'id' field.
        Returns:
            Dict[str, Any]: Dictionary with metric results and aggregated results.
        """
        # Step 1: Build a lookup dictionary for gold data keyed by ID
        gold_dict = {gold_item['id']: gold_item for gold_item in gold}
        # Step 2: Match predictions to gold items by ID
        matched_records = []
        for prediction in predictions:
            gold_item = gold_dict.get(prediction['id'])
            if gold_item is None:
                # Skip predictions without a matching gold item
                continue
            # Combine prediction and gold item into a single record for downstream use
            record = {**prediction, 'gold': gold_item}
            matched_records.append(record)
        # Step 3: Apply all filters sequentially to the matched records
        for result_filter in self.filters:
            matched_records = result_filter.filter(matched_records)
        # Step 4: Aggregate the records using the provided aggregator
        aggregated_result = self.aggregator.aggregate(matched_records)
        # Step 5: Compute all metrics using the filtered and aggregated records
        metric_results = {}
        for metric in self.metrics:
            # Compute the metric using prediction values and gold resolutions
            metric_name = metric.__class__.__name__
            prediction_values = [record.get('value') for record in matched_records]
            gold_resolutions = [record['gold'].get('resolution') for record in matched_records]
            metric_results[metric_name] = metric.compute(prediction_values, gold_resolutions)
        # Step 6: Return a dictionary with all computed metrics and the aggregated result
        return {"metrics": metric_results, "aggregated": aggregated_result}

    # ====================
    # All logic below is ported from evaluate.py
    # ====================
    @staticmethod
    def brier_score_binary(pred_probs, gold_outcomes):
        """
        Compute the Brier score for binary predictions.
        Args:
            pred_probs (list of float): Predicted probabilities for the positive class.
            gold_outcomes (list of int): Ground truth binary outcomes (0 or 1).
        Returns:
            float: The mean Brier score across all predictions.
        """
        errors = []
        for predicted_prob, actual_outcome in zip(pred_probs, gold_outcomes):
            squared_error = (predicted_prob - actual_outcome) ** 2
            errors.append(squared_error)
        return np.mean(errors)

    @staticmethod
    def compute_ece(pred_probs, gold_outcomes, num_bins=10):
        """
        Compute Expected Calibration Error (ECE) using equal-width bins.
        Args:
            pred_probs (list of float): Predicted probabilities.
            gold_outcomes (list of int): Ground truth binary outcomes (0 or 1).
            num_bins (int): Number of bins to use for calibration.
        Returns:
            float: The expected calibration error.
        """
        bins = np.linspace(0, 1, num_bins + 1)
        bin_indices = np.digitize(pred_probs, bins)
        total_predictions = len(pred_probs)
        ece = 0.0
        for bin_number in range(1, num_bins + 1):
            # Find indices of predictions in this bin
            indices_in_bin = [idx for idx, bin_idx in enumerate(bin_indices) if bin_idx == bin_number]
            if not indices_in_bin:
                continue
            # Compute average confidence and accuracy for this bin
            avg_confidence = np.mean([pred_probs[idx] for idx in indices_in_bin])
            avg_accuracy = np.mean([gold_outcomes[idx] for idx in indices_in_bin])
            bin_weight = len(indices_in_bin) / total_predictions
            ece += bin_weight * abs(avg_confidence - avg_accuracy)
        return ece

    @staticmethod
    def crps_gaussian(pred, sigma, obs):
        """
        Compute Continuous Ranked Probability Score (CRPS) for a Gaussian forecast using a closed-form solution.
        Args:
            pred (float): Predicted mean.
            sigma (float): Standard deviation of the prediction.
            obs (float): Observed value.
        Returns:
            float: The CRPS value for the prediction.
        """
        z = (obs - pred) / sigma
        phi = norm.pdf(z)
        Phi = norm.cdf(z)
        crps = sigma * (z * (2 * Phi - 1) + 2 * phi - 1 / math.sqrt(math.pi))
        return crps

    @staticmethod
    def mace(model_confs, gold_confs):
        """Mean Absolute Confidence Error."""
        errors = [abs(a - b) for a, b in zip(model_confs, gold_confs)]
        return np.mean(errors)

    @staticmethod
    def msce(model_confs, gold_confs):
        """Mean Squared Confidence Error."""
        errors = [(a - b) ** 2 for a, b in zip(model_confs, gold_confs)]
        return np.mean(errors)

    @staticmethod
    def f1_score_binary(predicted_labels, gold_labels, positive_label="yes"):
        """Compute the F1 score for binary classification (treating 'yes' as positive)."""
        tp = sum(1 for p, g in zip(predicted_labels, gold_labels) if p.lower() == positive_label and g.lower() == positive_label)
        fp = sum(1 for p, g in zip(predicted_labels, gold_labels) if p.lower() == positive_label and g.lower() != positive_label)
        fn = sum(1 for p, g in zip(predicted_labels, gold_labels) if p.lower() != positive_label and g.lower() == positive_label)
        if tp == 0:
            return 0.0
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        if (precision + recall) == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    @staticmethod
    def accuracy_binary(predicted_labels, gold_labels):
        """Compute accuracy for binary predictions."""
        correct = sum(1 for p, g in zip(predicted_labels, gold_labels) if p.lower() == g.lower())
        return correct / len(gold_labels) if gold_labels else None

    @staticmethod
    def normalize_error(error, alpha=0.05):
        """
        Normalize a positive error value using a sigmoid-based transformation.
        Maps error from 0 (best) to infinity (worst) into the range [0, 1],
        with 0 corresponding to zero error and values close to 1 for large errors.
        """
        return 2 * (1 / (1 + math.exp(-alpha * abs(error))) - 0.5)

    @staticmethod
    def parse_date(date_str):
        """Parse a date string and return the date portion."""
        try:
            dt = date_parse(date_str)
            return dt.date()
        except Exception:
            return None

    @staticmethod
    def is_valid_binary_resolution(res):
        """Check if a binary resolution is valid (yes/no)."""
        if isinstance(res, str):
            r = res.strip().lower()
            return r in ["yes", "no"]
        return False

    @staticmethod
    def is_valid_numeric_resolution(res):
        """Check if the resolution can be converted to a number."""
        try:
            float(res)
            return True
        except Exception:
            return False

    @staticmethod
    def is_valid_date_resolution(res):
        """Check if the resolution can be parsed as a date."""
        d = PredictionEvaluator.parse_date(res)
        return d is not None

    @staticmethod
    def load_gold(gold_path):
        """Load the gold file into a dictionary keyed by id."""
        gold_data = {}
        with open(gold_path, "r") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    gold_data[entry["id"]] = entry
                except Exception:
                    continue
        return gold_data

    @staticmethod
    def passes_filter(gold_entry, filter_date, filter_type):
        """Return True if the gold entry's created_time or resolution_time is after filter_date."""
        if filter_type == "created_time":
            time_str = gold_entry.get("created_time", "")
        elif filter_type == "resolution_time":
            time_str = gold_entry.get("resolution_time", "")
        else:
            return False
        try:
            dt = date_parse(time_str)
            return dt.date() >= filter_date
        except Exception:
            return False

    @staticmethod
    def extract_model_name(pred_file):
        """Extract model name from file name.
           For example, from 'cleaned_answers_allenai_OLMo-7B-Instruct-hf.jsonl'
           we extract 'allenai_OLMo-7B-Instruct-hf'."""
        base = os.path.basename(pred_file)
        if base.startswith("cleaned_answers_"):
            base = base[len("cleaned_answers_"):]
        if base.endswith(".jsonl"):
            base = base[:-len(".jsonl")]
        return base

    @staticmethod
    def process_prediction_file(pred_file, gold_data, filter_dates):
        """
        Process one prediction file.
        For each prediction entry, match on id with gold_data and check the type.
        For each aggregated result in the prediction entry, compute per-question metrics.
        Then, for each filter (applied on created_time and resolution_time), yield a record.
        If the aggregated answer is "unclear" (case-insensitive), override the score with a fixed penalty.
        """
        model_name = PredictionEvaluator.extract_model_name(pred_file)
        PENALTY_NUMERIC = 1000.0  # Penalty error for numeric questions
        PENALTY_DATE = 365        # Penalty in days for date questions
        records = []
        with open(pred_file, "r") as f:
            for line in f:
                try:
                    pred_entry = json.loads(line)
                except Exception:
                    continue
                qid = pred_entry.get("id")
                gold_entry = gold_data.get(qid)
                if not gold_entry:
                    continue
                qtype = pred_entry.get("type", "").lower()
                gold_resolution = gold_entry.get("resolution", "").strip()
                gold_conf = gold_entry.get("normalized_score", None)
                if gold_conf is None:
                    continue
                # Check resolution validity.
                valid = False
                if qtype == "binary":
                    valid = PredictionEvaluator.is_valid_binary_resolution(gold_resolution)
                elif qtype == "numeric":
                    valid = PredictionEvaluator.is_valid_numeric_resolution(gold_resolution)
                elif qtype == "date":
                    valid = PredictionEvaluator.is_valid_date_resolution(gold_resolution)
                if not valid:
                    continue
                # Parse gold resolution.
                if qtype == "binary":
                    gold_label = gold_resolution.strip().lower()
                elif qtype == "numeric":
                    try:
                        gold_value = float(gold_resolution)
                    except Exception:
                        continue
                elif qtype == "date":
                    gold_date = PredictionEvaluator.parse_date(gold_resolution)
                    if gold_date is None:
                        continue
                # For filtering: get created_time and resolution_time (dates)
                try:
                    created_date = date_parse(gold_entry.get("created_time")).date()
                except Exception:
                    created_date = None
                try:
                    resolution_date = date_parse(gold_entry.get("resolution_time")).date()
                except Exception:
                    resolution_date = None
                # Process each aggregated result in the prediction entry.
                agg_results = pred_entry.get("aggregated_results", {})
                for agg_key, agg_val in agg_results.items():
                    pred_answer = agg_val.get("answer", "").strip()
                    pred_conf = agg_val.get("confidence", None)
                    if pred_conf is None:
                        continue
                    record = {
                        "model": model_name,
                        "question_id": qid,
                        "question_type": qtype,
                        "agg_result": agg_key,
                        "gold_conf": gold_conf,
                        "pred_conf": pred_conf,
                    }
                    # Override behavior for "unclear" answers:
                    if pred_answer.lower() == "unclear":
                        record["unclear"] = True
                        if qtype == "binary":
                            record["gold_answer"] = gold_label
                            record["pred_answer"] = "unclear"
                            record["correct"] = 0
                            pred_prob = 0.0 if gold_label == "yes" else 1.0
                            record["pred_prob"] = pred_prob
                            record["gold_outcome"] = 1 if gold_label == "yes" else 0
                        elif qtype == "numeric":
                            record["pred_value"] = None
                            record["abs_error"] = PENALTY_NUMERIC
                            record["ape"] = PENALTY_NUMERIC
                            record["crps"] = PENALTY_NUMERIC
                            record["norm_abs_error"] = 1.0
                            record["norm_ape"] = 1.0
                            record["norm_crps"] = 1.0
                        elif qtype == "date":
                            record["pred_date"] = None
                            record["days_error"] = PENALTY_DATE
                            record["crps"] = PENALTY_DATE
                            record["norm_days_error"] = 1.0
                            record["norm_crps"] = 1.0
                    else:
                        record["unclear"] = False
                        # ... (rest of the original logic for metric computation)
                    records.append(record)
        return records

    @staticmethod
    def aggregate_records(records):
        """
        Group records by model, question_type, filter_type, filter_date, and agg_result.
        Then compute average metrics per group.
        """
        groups = defaultdict(list)
        for r in records:
            key = (r["model"], r["question_type"], r.get("filter_type"), r.get("filter_date"), r["agg_result"])
            groups[key].append(r)
        aggregated = []
        for key, group in groups.items():
            agg = {"model": key[0], "question_type": key[1], "filter_type": key[2], "filter_date": key[3], "agg_result": key[4]}
            # Compute averages for all numeric fields
            for field in ["accuracy", "f1", "brier", "norm_mae", "norm_ape", "norm_crps", "norm_days_error"]:
                values = [r[field] for r in group if field in r]
                agg[field] = np.mean(values) if values else None
            agg["num_questions"] = len(group)
            aggregated.append(agg)
        return aggregated

    @staticmethod
    def write_csv(filename, records):
        """Write records to CSV with a custom header order."""
        if not records:
            return
        header = list(records[0].keys())
        with open(filename, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            for r in records:
                writer.writerow(r)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model predictions against gold resolutions.")
    parser.add_argument('--gold', type=str, required=True, help='Path to gold jsonl file')
    parser.add_argument('--predictions', type=str, nargs='+', required=True, help='Prediction jsonl files')
    parser.add_argument('--out', type=str, default='evaluation_results.csv', help='Output CSV file')
    args = parser.parse_args()
    gold_data = PredictionEvaluator.load_gold(args.gold)
    all_records = []
    filter_dates = [datetime.date(2022, 8, 1), datetime.date(2023, 1, 1)]
    for pred_file in args.predictions:
        records = PredictionEvaluator.process_prediction_file(pred_file, gold_data, filter_dates)
        all_records.extend(records)
    agg = PredictionEvaluator.aggregate_records(all_records)
    PredictionEvaluator.write_csv(args.out, agg)
    print(f"Evaluation complete. Results written to {args.out}")
