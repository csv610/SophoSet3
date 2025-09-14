from typing import Dict, Any, List, Optional

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

from base_hf_dataset import BaseHFDataset, QAData

class MMLUProDataset(BaseHFDataset):
    DATASET_NAME = "TIGER-Lab/MMLU-Pro"
    
    def __init__(self):
        super().__init__(self.DATASET_NAME)
    
    def extract_row_data(self, row: Dict[str, Any], index: int) -> QAData:
        """Extract and format data from an MMLU-Pro dataset row.
        
        Args:
            row: The dataset row to extract data from
            index: The index of the row in the dataset
            
        Returns:
            QAData object containing the formatted row data
        """
        # Extract question, options, and answer
        question = row.get('question', '')
        options = row.get('options', [])
        answer = row.get('answer', '')
        
        return QAData(
            key=self.get_key(index),
            question=question,
            options=options,
            answer=answer
        )

if __name__ == "__main__":
    dset = MMLUProDataset()
    
    # Save to JSON and LMDB
    from dataset_exporter import DatasetExporter
    DatasetExporter.save(dset, format='lmdb')
