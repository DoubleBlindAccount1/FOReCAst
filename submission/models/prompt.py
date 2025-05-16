from abc import ABC, abstractmethod
from typing import Dict

class PromptTemplate(ABC):
    """
    Abstract base class for prompt templates. Subclasses must implement the format method.
    """
    @abstractmethod
    def format(self, question: str) -> str:
        """
        Format the prompt for the given question.
        Args:
            question (str): The question to format into the prompt.
        Returns:
            str: The formatted prompt string.
        """
        pass

class NumericPromptTemplate(PromptTemplate):
    """
    Prompt template for numeric questions. Instructs the AI to provide a numeric answer in JSON format.
    """
    def format(self, question: str) -> str:
        """
        Format the prompt for a numeric question.
        Args:
            question (str): The numeric question to format.
        Returns:
            str: The formatted prompt string for numeric questions.
        """
        return (
            """
            You are an AI assistant providing precise numerical forecasts. 
            Answer the following question with a single numeric value in JSON format.

            Example:
            Q: How much global photovoltaic energy generation was deployed by the end of 2020?  
            A: { \"value\": 738 }  

            Q: $question  
            A: { \"value\": """.replace("$question", question)
        )

class DatePromptTemplate(PromptTemplate):
    """
    Prompt template for date questions. Instructs the AI to provide a date answer in JSON format.
    """
    def format(self, question: str) -> str:
        """
        Format the prompt for a date question.
        Args:
            question (str): The date question to format.
        Returns:
            str: The formatted prompt string for date questions.
        """
        return (
            """
            You are an AI assistant providing precise date forecasts. 
            Answer the following question with a single date in YYYY-MM-DD format in JSON.

            Example:
            Q: When did an AI system achieve a significant victory against a professional human in Starcraft 2?  
            A: { \"value\": \"2019-01-24\" }

            Q: $question  
            A: { \"value\": """.replace("$question", question)
        )

class BinaryPromptTemplate(PromptTemplate):
    """
    Prompt template for binary (yes/no) questions. Instructs the AI to provide a binary answer in JSON format.
    """
    def format(self, question: str) -> str:
        """
        Format the prompt for a binary question.
        Args:
            question (str): The binary question to format.
        Returns:
            str: The formatted prompt string for binary questions.
        """
        return (
            """
            You are an AI assistant providing binary (Yes/No) answers. 
            Answer the following question with \"Yes\" or \"No\" in JSON format.

            Example:
            Q: Will we confirm evidence for megastructures orbiting the star KIC 8462852?  
            A: { \"value\": \"No\" }

            Q: $question  
            A: { \"value\": """.replace("$question", question)
        )

class PromptFactory:
    """
    Factory class for retrieving the appropriate PromptTemplate subclass based on question type.
    """
    _mapping = {
        "numeric": NumericPromptTemplate(),
        "date": DatePromptTemplate(),
        "binary": BinaryPromptTemplate(),
    }

    @staticmethod
    def get_prompt_template(qtype: str) -> PromptTemplate:
        """
        Retrieve the prompt template instance for the given question type.
        Args:
            qtype (str): The type of question ('numeric', 'date', or 'binary').
        Returns:
            PromptTemplate: The corresponding prompt template instance.
        Raises:
            KeyError: If the question type is not recognized.
        """
        if qtype not in PromptFactory._mapping:
            raise KeyError(f"Unknown question type: {qtype}")
        return PromptFactory._mapping[qtype]
