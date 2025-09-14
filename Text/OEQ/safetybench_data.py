from typing import Dict, Any, List, Optional

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

from base_hf_dataset import BaseHFDataset, QAData

class SafetyBenchDataset(BaseHFDataset):
    """A class to handle loading and managing the AIME 2025 dataset."""
    
    DATASET_NAME = "thu-coai/SafetyBench"
    
    def __init__(self):
        """Initialize the AIME 2025 dataset handler.
        """
        super().__init__(self.DATASET_NAME)
        
    def extract_row_data(self, row: Dict[str, Any], index: int) -> QAData:
        """Extract and format data from an AIME 2025 dataset row.
        
        Args:
            row: The dataset row to extract data from
            index: The index of the row in the dataset
            
        Returns:
            QAData object containing the formatted row data
        """
        # Extract question text
        question = row.get('question', '')
        answer = row.get('category', '')
        
        return QAData(
            key=self.get_key(index),
            question=question,
            answer=answer
        )

if __name__ == "__main__":
    dset = SafetyBenchDataset()
    
    from dataset_exporter import DatasetExporter
    DatasetExporter.save(dset, format='lmdb')
