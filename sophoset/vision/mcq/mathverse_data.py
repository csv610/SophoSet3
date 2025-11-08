from typing import Dict, Any, List, Optional, Literal

import sys
import os
import re

from sophoset.core.base_hf_dataset import BaseHFDataset, QAData

class MathVerseDataset(BaseHFDataset):
    """A class to handle loading and interacting with the MathVista dataset."""
    
    DATASET_NAME = "AI4Math/MathVerse"
    
    def __init__(self):
        """
        Initialize the dataset handler.
        """
        super().__init__(self.DATASET_NAME)

    def extract_question_and_options(self, text):
        """
        Extracts the question and multiple-choice options from a given text.

        Args:
            text (str): The input string containing the question and choices.

        Returns:
            dict: A dictionary with 'question' and 'options' keys. The 'question'
                  is a string and 'options' is a list of strings.
                  Returns None if the format is not recognized.
        """
        # Use a regex to find the question and the choices section.
        # The question is everything before 'Choices:' and the options are everything after.
        match = re.search(r'(.*?)\s*Choices:\s*\n(.*)', text, re.DOTALL | re.IGNORECASE)

        if not match:
            print("Error: 'Choices:' delimiter not found in the text.")
            return None

        # Extract the question part and clean up whitespace
        question_text = match.group(1).strip()
        
        # Extract the raw options text
        options_text = match.group(2).strip()

        # Split the options text by newline to get individual options
        # Then, clean up leading/trailing whitespace for each option
        options_list = [opt.strip() for opt in options_text.split('\n') if opt.strip()]

        # Return the extracted data in a dictionary
        return question_text, options_list

    
    def extract_row_data(self, row: Dict[str, Any], index: int) -> QAData:
        """
        Extract and format data from a dataset row.
        
        Args:
            row: The dataset row to extract data from
            index: The index of the row in the dataset
            
        Returns:
            QAData object containing the formatted row data
        """
        question = row.get('question', '')
        options = row.get('options', [])
        answer = row.get('answer', '')
        
        return QAData(
            key=self.get_key(index),
            question=question,
            options=options,
            answer=answer,
            image_path=row.get('image', '')
        )


if __name__ == "__main__":
    # Create the dataset
    dataset = MathVerseDataset()
    
    from sophoset.utils.dataset_exporter import DatasetExporter
    DatasetExporter.save(dataset, format='lmdb')
