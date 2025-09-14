from typing import Dict, Any, List, Optional
import re
import ast

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

from base_hf_dataset import BaseHFDataset, QAData

class MedicalMedowMedQADataset(BaseHFDataset):
    """A class to handle loading and managing the MedicalMeadowMedQA dataset."""
    
    DATASET_NAME = "medalpaca/medical_meadow_medqa"
    
    def __init__(self):
        """Initialize the MedicalMeadowMedQA dataset handler."""
        super().__init__(self.DATASET_NAME)
    
    def extract_row_data(self, row: Dict[str, Any], index: int) -> QAData:
        """Extract and format data from a MedicalMeadowMedQA dataset row.
        
        Args:
            row: The dataset row to extract data from.
            index: The index of the row in the dataset.
            
        Returns:
            QAData object containing the formatted row data.
        """
        input_text = row.get('input', '')
        
        question = ""
        options = []
        answer = ""
        
        # Use a non-greedy regular expression to find the last dictionary.
        # This is more robust to variations in trailing characters.
        match = re.search(r'(\{.*?\})[\s,]*$', input_text, re.DOTALL)
        
        if match:
            options_str = match.group(1).strip()
            
            try:
                # Safely parse the options string as a dictionary.
                options_dict = ast.literal_eval(options_str)
                
                # The question is everything before the options dictionary.
                question = input_text[:match.start()].strip()
                if question.startswith('Q:'):
                    question = question[2:].strip()
                options = list(options_dict.values())
            except (ValueError, SyntaxError):
                print(f"Error parsing options for row {index}. Skipping...")
                question = input_text
                options = []
        else:
            # Fallback if no dictionary is found.
            question = input_text
            options = []

        # The 'output' field contains the correct answer as a dictionary.
        output_dict = row.get('output', {})
        correct_answer_key = next(iter(output_dict), None)
        # Assign the key itself as the answer
        answer = correct_answer_key

        return QAData(
            key=self.get_key(index),
            question=question,
            options=options,
            answer=answer
        )

if __name__ == "__main__":
    dset = MedicalMedowMedQADataset()
    
    from dataset_exporter import DatasetExporter
    DatasetExporter.save(dset, format='lmdb')
