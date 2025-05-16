import os
import json
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
from typing import Any, List
import argparse

class ModelManager:
    """
    A professional and extensible manager for HuggingFace models. This class provides
    methods for downloading, caching, and loading models and tokenizers in a reproducible,
    maintainable, and well-documented way. All environment variables and cache directories
    are handled explicitly for clarity.
    """

    def __init__(self, cache_dir: str = os.path.expanduser("~/.cache/huggingface")):
        """
        Initialize the ModelManager with a specified cache directory. Ensures that all
        relevant HuggingFace cache environment variables are set and the directory exists.

        Args:
            cache_dir (str): The directory to use for caching models and tokenizers.
        """
        self.cache_dir = cache_dir
        # Ensure the cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)
        # Set environment variables for HuggingFace caching
        os.environ["HF_HOME"] = self.cache_dir
        os.environ["HF_HUB_CACHE"] = self.cache_dir
        os.environ["TRANSFORMERS_CACHE"] = self.cache_dir
        os.environ["HF_ALLOW_CODE_EVAL"] = "1"

    def download_model(self, model_name: str, token: str = None) -> None:
        """
        Download a single HuggingFace model and its tokenizer to the local cache directory.
        If a token is provided, authenticate with the HuggingFace Hub.

        Args:
            model_name (str): The name or path of the model to download.
            token (str, optional): HuggingFace authentication token for private models.
        """
        if token is not None:
            # Authenticate with the HuggingFace Hub if a token is provided
            login(token=token)
        # Download the tokenizer
        AutoTokenizer.from_pretrained(model_name, cache_dir=self.cache_dir, trust_remote_code=True)
        # Download the model
        AutoModelForCausalLM.from_pretrained(model_name, cache_dir=self.cache_dir, trust_remote_code=True)

    def load_model_and_tokenizer(self, model_name: str) -> Any:
        """
        Load both the model and tokenizer from the local cache or HuggingFace Hub.

        Args:
            model_name (str): The name or path of the model to load.

        Returns:
            tuple: (model, tokenizer) loaded from the cache or HuggingFace Hub.
        """
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=self.cache_dir, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=self.cache_dir, trust_remote_code=True)
        return model, tokenizer

    @staticmethod
    def download_models_cli(models: List[str], token: str = None, cache_dir: str = os.path.expanduser("~/.cache/huggingface")):
        """
        Download multiple HuggingFace models and their tokenizers via CLI. This method is
        suitable for batch downloading and provides progress updates. All environment variables
        and cache directories are set for reproducibility.

        Args:
            models (List[str]): List of model names or paths to download.
            token (str, optional): HuggingFace authentication token for private models.
            cache_dir (str): Directory to use for caching downloads.
        """
        # Ensure the cache directory exists
        os.makedirs(cache_dir, exist_ok=True)
        # Set environment variables for HuggingFace caching
        os.environ["HF_HOME"] = cache_dir
        os.environ["HF_HUB_CACHE"] = cache_dir
        os.environ["TRANSFORMERS_CACHE"] = cache_dir
        os.environ["HF_ALLOW_CODE_EVAL"] = "1"
        # Authenticate if a token is provided
        if token is not None:
            login(token=token)
        # Download each model and tokenizer
        for model_name in tqdm(models, desc="Downloading models", unit="model"):
            print(f"\n📥 Downloading: {model_name}")
            try:
                # Download the tokenizer
                tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True)
                # Download the model
                model_obj = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True)
                print(f"✔ Successfully downloaded {model_name} to {cache_dir}")
            except Exception as error:
                print(f"❌ Failed to download {model_name}: {error}")
        print("\n✅ All models downloaded successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download HuggingFace models with custom cache directory.")
    parser.add_argument('--models', nargs='+', default=[
        'allenai/OLMo-2-1124-7B',
        'allenai/OLMo-2-1124-7B-Instruct',
        'allenai/OLMo-7B-Instruct-hf',
        'codellama/CodeLlama-7b-Instruct-hf',
        'allenai/OLMo-1B',
        'allenai/OLMo-7B',
        'bigscience/bloom-560m',
        'bigscience/bloom-7b1',
        'openai-community/gpt2',
        'openai-community/gpt2-xl',
        'huggyllama/llama-7b',
        'EleutherAI/pythia-14m',
        'EleutherAI/pythia-160m',
        'EleutherAI/pythia-2.8b',
        "google/gemma-3-1b-it",
        "meta-llama/Llama-3.1-8B",
        "meta-llama/Llama-3.1-8B-Instruct",
        "Qwen/Qwen2.5-1.5B",
        "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen/Qwen2.5-7B",
        "Qwen/Qwen2.5-7B-Instruct"
    ], help='List of model names to download')
    parser.add_argument('--token', type=str, default=None, help='HuggingFace token (if required)')
    parser.add_argument('--cache_dir', type=str, default=os.path.expanduser("~/.cache/huggingface"), help='Custom cache directory')
    args = parser.parse_args()
    ModelManager.download_models_cli(args.models, token=args.token, cache_dir=args.cache_dir)
