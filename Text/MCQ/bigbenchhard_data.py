from typing import Dict, Any, Optional
import re

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

from base_hf_dataset import BaseHFDataset, QAData

class BigBenchHardDataset(BaseHFDataset):
    """A class to handle loading and managing the BigBenchHard dataset without UI dependencies."""
    
    DATASET_NAME = "Joschka/big_bench_hard"

    def __init__(self):
        """Initialize the base class ."""
        super().__init__(self.DATASET_NAME)
    
    def extract_row_data(self, row: Dict[str, Any], index: int) -> 'QAData':
        """Extract the data from the row.
        
        Args:
            row: The dataset row to extract data from
            index: The index of the row in the dataset
            
        Returns:
            QAData object containing the formatted row data
        """
        question = row.get('question', "")
        # Handle options which is a dictionary with 'label' and 'text' keys
        options = []
        for choice in row.get('choices', []):
            if isinstance(choice, dict):
                options.append(choice.get('text', ''))
            else:
                options.append(str(choice))
            
        answer = row.get('target', "")
        
        return QAData(
            key=self.get_key(index),
            question=question, 
            options=options,
            answer=answer
        )

if __name__ == "__main__":
    dset = BigBenchHardDataset()
    
    from dataset_exporter import DatasetExporter
    DatasetExporter.save(dset, format='lmdb')
