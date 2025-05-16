# FOReCAst: The Future Outcome Reasoning and Confidence Assessment Benchmark

---

## Summary

FOReCAst is a [anonymized for double blind reviewing].

---

## Architecture Overview

The FOReCAst pipeline is composed of the following core layers:

- **Data Layer**: Responsible for ingestion, cleaning, normalization, and splitting of datasets. Implements abstract base classes and concrete loaders for JSONL and other formats. Includes gold-standard data processing and question extraction utilities.
- **Model Layer**: Encapsulates model management, downloading, caching, and inference. Supports extensible prompt templates and model runners. All model interactions are abstracted via dedicated manager and inference engine classes.
- **Evaluation Layer**: Defines metrics (accuracy, F1, Brier, CRPS, etc.), aggregation strategies (majority vote, weighted average, etc.), and filtering mechanisms. Evaluation is orchestrated through an extensible evaluator class.
- **Postprocessing Layer**: Handles output parsing, cleaning, and LaTeX table generation. Ensures outputs are normalized and publication-ready.
- **Utility Layer**: Provides centralized configuration and a robust, timestamped logging utility for all pipeline stages.

---

## Directory Structure

```
submission/
  data/
    cleaner.py           # DataCleaner: static text/binary/numeric/date cleaning
    dataset_splitter.py  # DatasetSplitter: stratified data splitting
    gold_processor.py    # GoldDataProcessor: gold standard normalization
    gold_question_processor.py
    gold_splitter.py
    loader.py            # BaseDataLoader/JsonlDataLoader: extensible data ingestion
    question_processor.py
    splitter.py          # DataSplitter: type-preserving splits
  models/
    inference.py         # InferenceEngine: model inference logic
    inference_runner.py
    manager.py           # ModelManager: download/load models
    prompt.py            # PromptTemplate/PromptFactory: prompt OOP
    __init__.py
  evaluation/
    aggregator.py        # Aggregator/strategies: majority, weighted, etc.
    evaluator.py         # PredictionEvaluator: orchestrates evaluation
    filters.py           # ResultFilter/CutoffDateFilter/ModelTypeFilter
    metrics.py           # Metric/Accuracy/F1/Brier/CRPS
  postprocessing/
    aggregation_results_filter.py
    answer_postprocessor.py
    cutoff_match_results_filter.py
    cutoff_results_filter.py
    latex_table.py       # LatexTableGenerator: publication tables
    output_parser.py     # OutputParser: value/confidence extraction
    parser.py
    question_stats.py
    cleaner.py           # OutputCleaner: type-based cleaning
  utils/
    config.py            # Config: centralized settings
    logger.py            # Logger: timestamped, leveled logging
  main.py                # Pipeline entry point
  README.md              # This file
```

---

## Main Pipeline Flow

1. **Argument Parsing**: All CLI options are parsed using argparse, with defaults pulled from `Config`.
2. **Data Loading & Cleaning**: Data is loaded using a `JsonlDataLoader` and cleaned with `DataCleaner`, ensuring all fields are normalized.
3. **Data Splitting**: The `DataSplitter` performs stratified splits by type, ensuring reproducibility via random seed.
4. **Gold Data Processing**: Gold standards are normalized via `GoldDataProcessor`.
5. **Model Management**: `ModelManager` downloads and loads the requested model and tokenizer, with caching.
6. **Inference**: The `InferenceEngine` runs inference on the test set, producing structured predictions.
7. **Postprocessing**: Outputs are parsed (`OutputParser`), cleaned (`OutputCleaner`), and normalized.
8. **Evaluation**: Predictions are evaluated using a configurable list of metrics, aggregators, and filters via `PredictionEvaluator`.
9. **Results Generation**: Metrics are logged, and results are rendered as LaTeX tables using `LatexTableGenerator`.
10. **Logging**: All steps are logged with timestamps and severity via `Logger`.

---

## Extensibility & Customization

### Adding New Metrics
- Inherit from `Metric` (see `evaluation/metrics.py`).
- Implement the `compute` method with your custom logic.
- Register your metric in the pipeline as needed.

### Adding New Aggregators
- Inherit from `Aggregator` (see `evaluation/aggregator.py`).
- Implement the `aggregate` method.
- Add to the evaluation pipeline.

### Adding New Filters
- Inherit from `ResultFilter` (see `evaluation/filters.py`).
- Implement the `filter` method for custom filtering logic.

### Adding New Prompt Templates
- Inherit from `PromptTemplate` (see `models/prompt.py`).
- Register with `PromptFactory` for dynamic prompt instantiation.

### Adding New Model Runners
- Extend `ModelManager` and/or `InferenceEngine` for new model types or inference logic.

---

## Configuration
- All global paths, filenames, and parameters are centralized in `utils/config.py` (`Config` class).
- Update device, random seed, and directory settings as needed.

## Logging
- Use `Logger` for all console output. Supports info, warning, and error levels with ISO timestamps.

## Error Handling
- All file operations, parsing, and model interactions are wrapped in robust error handling blocks.
- Errors are logged and, where recoverable, skipped with warnings.

---

## Requirements
- Python 3.8+
- `numpy`, `pandas`, `scipy`, `huggingface` (transformers), and other dependencies as listed in `requirements.txt`.

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Usage

### CLI

```bash
python -m submission.main \
  --train path/to/train.jsonl \
  --dev path/to/dev.jsonl \
  --test path/to/test.jsonl \
  --gold path/to/gold.jsonl \
  --model meta-llama/Llama-3.1-8B \
  --train_ratio 0.65 \
  --dev_ratio 0.1 \
  --test_ratio 0.25 \
  --output_table results_table.tex \
  --caption "Results Table" \
  --label "tab:results"
```

All arguments are optional and default to values in `Config`.
