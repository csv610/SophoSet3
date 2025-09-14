from typing import Dict, Any, List, Optional

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

from base_hf_dataset import BaseHFDataset, QAData

class Gsm8kDataset(BaseHFDataset):
    """A class to handle loading and managing the GSM8K dataset."""
    
    DATASET_NAME = "openai/gsm8k"
    
    def __init__(self):
        """Initialize the GSM8K dataset handler.
        """
        super().__init__(self.DATASET_NAME)
    
    def extract_row_data(self, row: Dict[str, Any], index: int) -> QAData:
        """Extract and format data from a GSM8K dataset row.
        
        Args:
            row: The dataset row to extract data from
            index: The index of the row in the dataset
            
        Returns:
            QAData object containing the formatted row data
        """
        # Extract question and answer
        question = row.get('question', '')
        answer = row.get('answer', '')
        
        # For GSM8K, the answer typically includes the reasoning steps
        # We'll extract just the final answer for the answer field
        final_answer = ''
        if answer and '####' in answer:
            final_answer = answer.split('####')[-1].strip()
        
        return QAData(
            key=self.get_key(index),
            question=question,
            answer=final_answer,
            explanation = answer
        )

if __name__ == "__main__":
    dset = Gsm8kDataset()
    
    from dataset_exporter import DatasetExporter
    DatasetExporter.save(dset, format='lmdb')
