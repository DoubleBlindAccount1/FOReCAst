import os
import json
import torch
import argparse
import re
import torch.nn.functional as F
from tqdm import tqdm
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from string import Template
import numpy as np

class InferenceRunner:
    """
    A professional, extensible runner for model inference and result caching.
    This class loads a model and tokenizer, processes questions, manages caching, and runs inference with detailed logging and documentation.
    """
    def __init__(self, model_name: str, question_file: str = "test.jsonl", cache_dir: str = None):
        """
        Initialize the InferenceRunner.
        Args:
            model_name (str): The name of the model to use for inference.
            question_file (str): Path to the JSONL file with questions.
            cache_dir (str, optional): Directory to use for Hugging Face cache. Defaults to ~/.cache/huggingface.
        """
        self.model_name = model_name
        self.question_file = question_file
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/huggingface")
        # Set Hugging Face environment variables for cache management
        os.environ["HF_HOME"] = self.cache_dir
        os.environ["HF_HUB_CACHE"] = self.cache_dir
        os.environ["TRANSFORMERS_CACHE"] = self.cache_dir
        os.environ["HF_ALLOW_CODE_EVAL"] = "1"
        os.makedirs(self.cache_dir, exist_ok=True)
        # Define the cache file for storing answers
        self.cache_file = f"cached_answers_{model_name.replace('/', '_')}.jsonl"
        # Set device for inference
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        # Load questions and cached results
        self.questions = self.load_questions()
        self.cached_results = self.load_cache()

    def load_questions(self) -> list:
        """
        Load questions from the specified JSONL file.
        Returns:
            list: List of question dictionaries.
        """
        questions = []
        with open(self.question_file, "r", encoding="utf-8") as f:
            for line in f:
                questions.append(json.loads(line))
        return questions

    def load_cache(self) -> dict:
        """
        Load cached results from the cache file if it exists.
        Returns:
            dict: Mapping from question ID to cached result.
        """
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "r", encoding="utf-8") as f:
                return {json.loads(line)["id"]: json.loads(line) for line in f}
        return {}

    def get_prompt_templates(self) -> dict:
        """
        Get the prompt templates for the current model. For 'Instruct' models, uses detailed templates with examples.
        Returns:
            dict: Mapping from question type to string.Template object.
        """
        if "Instruct" in self.model_name:
            return {
                "numeric": Template("""
        You are an AI assistant providing precise numerical forecasts. 
        Answer the following question with a single numeric value in JSON format.

        Example:
        Q: How much global photovoltaic energy generation was deployed by the end of 2020?  
        A: { \"value\": 738 }  

        Q: $question  
        A: { \"value\": """),
                "date": Template("""
        You are an AI assistant providing precise date forecasts. 
        Answer the following question with a single date in YYYY-MM-DD format in JSON.

        Example:
        Q: When did an AI system achieve a significant victory against a professional human in Starcraft 2?  
        A: { \"value\": \"2019-01-24\" }

        Q: $question  
        A: { \"value\": """),
                "binary": Template("""
        You are an AI assistant providing binary (Yes/No) answers. 
        Answer the following question with \"Yes\" or \"No\" in JSON format.

        Example:
        Q: Will we confirm evidence for megastructures orbiting the star KIC 8462852?  
        A: { \"value\": \"No\" }

        Q: $question  
        A: { \"value\": """),
            }
        else:
            # Placeholder for non-Instruct model templates
            return {
                "numeric": Template("""
        Q: How much global photovoltaic energy generation was deployed by the end of 2020?  
        A: { \"value\": 738 }  

        Q: $question  
        A: { \"value\": """),
                "date": Template("""
        Q: When did an AI system achieve a significant victory against a professional human in Starcraft 2?  
        A: { \"value\": \"2019-01-24\" }

        Q: $question  
        A: { \"value\": """),
                "binary": Template("""
        Q: Will we confirm evidence for megastructures orbiting the star KIC 8462852?  
        A: { \"value\": \"No\" }

        Q: $question  
        A: { \"value\": """),
            }

    def setup_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.cache_dir, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, cache_dir=self.cache_dir, trust_remote_code=True).to(self.device)
        self.pipeline = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, device=0 if self.device.type == "cuda" else -1)

    @staticmethod
    def compute_confidences(logits, generated_tokens):
        # Compute confidence scores for generated tokens
        probs = F.softmax(torch.tensor(logits), dim=-1)
        confidences = [float(probs[i, tok]) for i, tok in enumerate(generated_tokens)]
        return np.mean(confidences)

    def run(self):
        self.setup_model()
        prompt_templates = self.get_prompt_templates()
        results = []
        for q in tqdm(self.questions, desc=f"Generating answers with {self.model_name}", unit="question"):
            question_id = q["id"]
            if question_id in self.cached_results:
                results.append(self.cached_results[question_id])
                continue
            qtype = q.get("type", "binary")
            prompt = prompt_templates[qtype].substitute(question=q["question"])
            output = self.pipeline(prompt, max_new_tokens=32, return_full_text=False)[0]["generated_text"]
            # Extract answer from model output
            match = re.search(r'\{\s*"value"\s*:\s*("[^"]*"|\d+(?:\.\d+)?)\s*\}', output)
            answer = match.group(1).strip('"') if match else None
            result = {"id": question_id, "question": q["question"], "type": qtype, "answer": answer, "raw_output": output}
            results.append(result)
            # Cache result
            with open(self.cache_file, "a") as f:
                f.write(json.dumps(result) + "\n")
        return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on a specific model and question set.")
    parser.add_argument("--model", type=str, required=True, help="Model name to use for inference")
    parser.add_argument("--question_file", type=str, default="test.jsonl", help="Question JSONL file")
    parser.add_argument("--cache_dir", type=str, default=None, help="Custom Hugging Face cache directory")
    args = parser.parse_args()
    runner = InferenceRunner(args.model, args.question_file, args.cache_dir)
    runner.run()
