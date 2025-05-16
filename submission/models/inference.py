import torch
from typing import List, Dict, Any
from .prompt import PromptFactory

class InferenceEngine:
    """
    A professional, extensible engine for running inference using a Hugging Face model and tokenizer.
    This class takes a list of question dictionaries, formats prompts using PromptFactory, and generates model outputs.
    Extensive comments and explicit logic are included for maintainability and clarity.
    """
    def __init__(self, model, tokenizer, device: str = None):
        """
        Initialize the InferenceEngine.
        Args:
            model: The Hugging Face model used for inference.
            tokenizer: The tokenizer associated with the model.
            device (str, optional): Device to run inference on ('cuda' or 'cpu'). If not specified, will auto-detect.
        """
        self.model = model
        self.tokenizer = tokenizer
        # Automatically select CUDA if available, otherwise CPU
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Move the model to the target device
        self.model.to(self.device)

    def infer(self, questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Run inference on a list of questions, generating model outputs for each.
        Args:
            questions (List[Dict[str, Any]]): List of question dictionaries, each containing 'id', 'question', and 'type'.
        Returns:
            List[Dict[str, Any]]: List of result dictionaries with model outputs.
        """
        results = []  # List to hold the inference results
        for question_dict in questions:
            # Extract question type (e.g., 'numeric', 'date', or 'binary')
            question_type = question_dict.get("type")
            # Get the appropriate prompt template for this question type
            prompt_template = PromptFactory.get_prompt_template(question_type)
            # Format the prompt with the actual question
            formatted_prompt = prompt_template.format(question_dict["question"])
            # Tokenize the prompt for the model
            tokenized_inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
            # Generate output using the model (disable gradients for efficiency)
            with torch.no_grad():
                model_outputs = self.model.generate(**tokenized_inputs, max_new_tokens=64)
            # Decode the generated tokens into a string
            generated_text = self.tokenizer.decode(model_outputs[0], skip_special_tokens=True)
            # Append the result dictionary for this question
            results.append({
                "id": question_dict["id"],
                "question": question_dict["question"],
                "type": question_type,
                "generated": generated_text
            })
        return results
