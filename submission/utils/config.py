class Config:
    """
    Centralized configuration class for all major project paths, filenames, and global parameters.
    Designed for clarity, maintainability, and extensibility. All configuration values are class attributes.
    """
    # Data directory paths
    DATA_DIR = "data"  # Directory for data files
    MODELS_DIR = "models"  # Directory for model files
    EVAL_DIR = "evaluation"  # Directory for evaluation scripts and results
    POSTPROCESS_DIR = "postprocessing"  # Directory for postprocessing scripts
    UTILS_DIR = "utils"  # Directory for utility scripts
    CACHE_DIR = "~/.cache/huggingface"  # Directory for Hugging Face cache

    # Standard file names
    TRAIN_FILE = "train.jsonl"  # Training set file
    DEV_FILE = "dev.jsonl"  # Development set file
    TEST_FILE = "test.jsonl"  # Test set file
    GOLD_FILE = "gold.jsonl"  # Gold standard file

    # Miscellaneous global parameters
    RANDOM_SEED = 42  # Default random seed for reproducibility
    DEFAULT_DEVICE = "cuda"  # Default device for computation
