from typing import Dict, Any, List, Optional, Literal

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

from base_hf_dataset import BaseHFDataset, QAData

class MathVistaDataset(BaseHFDataset):
    """A class to handle loading and interacting with the MathVista dataset."""
    
    DATASET_NAME = "AI4Math/MathVista"
    
    def __init__(self, split: str = "test"):
        """
        Initialize the dataset handler.
        
        Args:
            split: The data split to use (default is 'test' as it's the main evaluation split)
        """
        super().__init__(self.DATASET_NAME)
        self.split = split
    
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
    dataset = MathVistaDataset(split="test")
    
    from dataset_exporter import DatasetExporter
    DatasetExporter.save(dataset, format='lmdb')
