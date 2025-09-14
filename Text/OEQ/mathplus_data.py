from typing import Dict, Any, List, Optional

import re
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

from base_hf_dataset import BaseHFDataset, QAData

class MathPlusDataset(BaseHFDataset):
    """A class to handle loading and managing the MATH-Plus dataset."""
    
    DATASET_NAME = "TIGER-Lab/MATH-plus"
    
    def __init__(self):
        """Initialize the base class ."""
        super().__init__(self.DATASET_NAME)

    def extract_short_answer(self, text):
        """
        Args:
            text (str): The input string containing the LaTeX code.
        Returns:
            str or None: The extracted value if found, otherwise None.
        """
        answer_pattern = r"The answer is (.*)"
        match = re.search(answer_pattern, text)
        if match:
           return match.group(1).strip(".")
        
        return None

    def extract_row_data(self, row: Dict[str, Any], index: int) -> QAData:
        """Extract and format data from the dataset row.
        
        Args:
            row: The dataset row to extract data from
            index: The index of the row in the dataset
            
        Returns:
            QAData object containing the formatted row data
        """
        # Extract question and solution
        question = row.get('instruction', '')
        explanation   = row.get('output', '')
        answer        = self.extract_short_answer(explanation)      
        
        return QAData(
            key=self.get_key(index),
            question=question,
            answer=answer,
            explanation= explanation
        )

if __name__ == "__main__":
    dset = MathPlusDataset()
    
    from dataset_exporter import DatasetExporter
    DatasetExporter.save(dset, format='lmdb')
