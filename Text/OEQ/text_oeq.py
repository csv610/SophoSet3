from any_llm import completion
import os
from pydantic import BaseModel
from typing import List, Dict, Any
import json
from pathlib import Path
from datetime import datetime

class PromptGenerator:
    """
    A class to handle the generation of open-ended question prompts
    for an LLM, with a focus on detailed and thoughtful responses.
    """
    def generate_prompt(self, question: str) -> str:
        """
        Generates an open-ended question prompt for an LLM.
        The prompt is designed to elicit detailed, well-reasoned responses.
        """
        # Task
        task_prompt = f"""Please provide a detailed and thoughtful response to the following question:

Question: {question}

Guidelines for your response:
1. Be comprehensive and cover all relevant aspects of the question
2. Provide clear explanations and reasoning
3. Include examples or evidence where appropriate
4. Structure your response in clear, coherent paragraphs
5. If the question is complex, break down your answer into logical sections

Your response:
"""
        return task_prompt

class TextOEQ:
    """
    A class to handle the generation of multiple-choice questions,
    communication with an LLM, and parsing of the structured response.
    """
    def __init__(self, model: str, provider: str, temperature: float = 0.0):
        self.model = model
        self.provider = provider
        self.temperature = temperature
        self.system_prompt = ""
        self.prompt_generator = PromptGenerator()
        
    def set_system_prompt(self, new_prompt: str):
        """
        Sets a new system prompt for the LLM.
        """
        self.system_prompt = new_prompt

    def get_response(self, question: str) -> str:
        """
        Asks an open-ended question to an LLM via the OpenRouter provider
        and returns the text response.
        """
        if not question:
            raise ValueError("The 'question' cannot be empty.")
            
        prompt = self.prompt_generator.generate_prompt(question)

        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        
        messages.append({"role": "user", "content": prompt})

        response = completion(
            model=self.model,
            provider=self.provider,
            messages=messages,
            temperature=self.temperature
        )

        result = response.choices[0].message.content
        return result

if __name__ == "__main__":
    # Make sure you have the appropriate environment variable set
    assert os.environ.get('OPENROUTER_API_KEY')

    # Create an instance of the TextMCQ with a custom temperature and system prompt
    textoeq = TextOEQ(
        model="deepseek/deepseek-chat-v3.1:free",
        provider="openrouter"
    )

    question = "What is theory of General Relativity?"
    
    try:
        # Get the response from the LLM
        response = textoeq.get_response(question)
        
    except ValueError as e:
        print(f"\nError processing LLM response: {e}")

