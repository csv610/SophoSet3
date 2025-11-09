import os
from litellm import completion
from pydantic import BaseModel
from typing import List

class Explanation(BaseModel):
    option: str
    is_correct: bool
    text: str

class ModelResponse(BaseModel):
    answer: str
    explanations: List[Explanation]

class PromptGenerator:
    """
    A class to handle the generation of multiple-choice question prompts
    for an LLM, adhering to a specific format.
    """
    def generate_prompt(self, question: str, options: List[str]) -> str:
        """
        Generates a multiple-choice question prompt for an LLM.
        The prompt is broken down into four parts: Persona, Task, Output Format, and Specific Instructions.
        """
        
        # 1. Task
        task_prompt = f"Answer the following multiple-choice question.\n\nQuestion: {question}\n\nOptions:\n"
        for i, option in enumerate(options):
            # Using A, B, C, D, E... for options
            task_prompt += f"{chr(65 + i)}. {option}\n"

        # 2. Output Format
        output_format_prompt = """
Please provide the correct option and a detailed explanation for each choice, formatted as follows:

Answer: 

Explanations:

"""
        for i in range(len(options)):
            option_letter = chr(65 + i)
            output_format_prompt += f"{option_letter}. True/False: [Your explanation for Option {option_letter}]\n"
        
        # Example part remains static and provides a clear template.
        output_format_prompt += """
Example:

Question: What is the primary function of insulin in the human body?
Options:
A. Producing red blood cells
B. Regulating blood glucose levels
C. Digesting proteins in the stomach
D. Storing fat in adipose tissue

Answer: B

Explanations:

A. False: red blood cells are produced in the bone marrow.
B. True:  insulin's primary role is to help cells absorb glucose from the bloodstream, thereby lowering blood sugar.
C. False: proteins are digested by enzymes like pepsin in the stomach.
D. False: while insulin helps in fat storage, its primary function is glucose regulation, not just fat storage.
"""
        
        # 3. Specific Instructions
        specific_instructions_prompt = """
Your response for the question above should start with the correct option letter, followed by your explanations for all options in the format shown."""

        full_prompt = f"{task_prompt}\n\n{output_format_prompt}\n\n{specific_instructions_prompt}"
        return full_prompt.strip()

class TextMCQ:
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

    def get_response(self, question: str, options: List[str]) -> ModelResponse:
        """
        Asks a text-based multiple-choice question to an LLM using litellm
        and returns a structured ModelResponse with response validation via Pydantic.
        """
        if not question:
            raise ValueError("The 'question' cannot be empty.")
        if not isinstance(options, list) or len(options) <= 1:
            raise ValueError("The 'options' must be a list with more than one element.")

        user_prompt = self.prompt_generator.generate_prompt(question, options)

        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        messages.append({"role": "user", "content": user_prompt})

        response = completion(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            response_format=ModelResponse
        )

        # response_format returns the parsed object directly for Pydantic models
        if hasattr(response, 'parsed'):
            return response.parsed
        else:
            return response


if __name__ == "__main__":
    # Make sure you have the appropriate environment variables set
    # For OpenAI models:
    assert os.environ.get('OPENAI_API_KEY') or os.environ.get('OPENROUTER_API_KEY')

    # Create an instance of the TextMCQ
    # Note: response_format requires models that support structured output (e.g., gpt-4o, gpt-4-turbo, etc.)
    textmcq = TextMCQ(
        model="gpt-4o-2024-08-06",
        provider="openai"
    )

    question = "Which of the following is an example of a renewable energy source?"
    options = ["Coal", "Natural Gas", "Solar Power", "Nuclear Energy", "Petroleum"]

    try:
        # Get the parsed Pydantic response from the LLM with structured output
        response = textmcq.get_response(question, options)

        print(f"Question: {question}")
        print(f"Options: {options}")

        print("\nParsed Pydantic Response:")
        print(f"Correct Answer: {response.answer}")
        print("\nExplanations:")
        for explanation in response.explanations:
            print(f"Option {explanation.option}: Correct? {explanation.is_correct}, Explanation: {explanation.text}")
    except Exception as e:
        print(f"\nError: {e}")

