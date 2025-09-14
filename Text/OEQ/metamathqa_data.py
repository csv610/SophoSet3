from typing import Dict, Any, Optional

import re
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

from base_hf_dataset import BaseHFDataset, QAData

class MetaMathQADataset(BaseHFDataset):
    """A class to handle loading and managing the MetaMathQA dataset."""
    
    DATASET_NAME = "meta-math/MetaMathQA"
    
    def __init__(self):
        """Initialize the MetaMathQA dataset handler."""
        super().__init__(self.DATASET_NAME)

    def extract_final_answer(self, text: str) -> str | None:
        """
        Extracts the complete string after 'The answer is: ' in the text.

        Args:
            text: The input string containing the final answer.

        Returns:
            The extracted answer string after 'The answer is: ', or None if not found.
        """
        # Find the position of 'The answer is: ' in the text
        prefix = "The answer is: "
        start_idx = text.find(prefix)
        
        # If the prefix is found, return everything after it
        if start_idx != -1:
            return text[start_idx + len(prefix):].strip()
        
        # Return None if the prefix is not found
        return None 


    def extract_row_data(self, row: Dict[str, Any], index: int) -> QAData:
        """Extract and format data from a MetaMathQA dataset row.
        
        Args:
            row: The dataset row to extract data from
            index: The index of the row in the dataset
            
        Returns:
            QAData object containing the formatted row data
        """
        # Extract question and solution
        question = row.get('query', '')
        explanation = row.get('response', '')
        answer      = self.extract_final_answer(explanation)
        
        return QAData(
            key=self.get_key(index),
            question=question,
            answer=answer,
            explanation=explanation
        )

if __name__ == "__main__":
    dset = MetaMathQADataset()
    
    from dataset_exporter import DatasetExporter
    DatasetExporter.save(dset, format='lmdb')
