from typing import Dict, Any, Optional, List

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

from base_hf_dataset import BaseHFDataset, QAData
from dataset_exporter import DatasetExporter

class Ai2ArcDataset(BaseHFDataset):
    """A class to handle loading and managing the AI2 ARC dataset without UI dependencies."""
    
    DATASET_NAME = "allenai/ai2_arc"
    
    def __init__(self):
        """Initialize the AI2 ARC dataset handler."""
        super().__init__(self.DATASET_NAME)
        
    def extract_row_data(self, row: Dict[str, Any], index: int) -> 'QAData':
        """Extract and format data from an AI2 ARC dataset row.
        
        Args:
            row: The dataset row to extract data fromls
            index: The index of the row in the dataset
            
        Returns:
            QAData object containing the formatted row data
        """
        question = row.get('question', '')
        choices = row.get('choices', {})
        options = choices.get('text', []) if isinstance(choices, dict) else []
        answer  = row.get('answerKey', '')
        
        return QAData(
            key=self.get_key(index),
            question=question,
            options=options,
            answer=answer
        )

if __name__ == "__main__":
    dset = Ai2ArcDataset()
    DatasetExporter.save(dset, format='lmdb')
