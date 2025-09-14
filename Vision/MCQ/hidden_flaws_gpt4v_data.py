"""
Hidden Flaws GPT-4V Dataset Handler

This module provides functionality to load and interact with the Hidden Flaws GPT-4V dataset.
It encapsulates all dataset-related operations, separating them from the UI code.
"""
from typing import Dict, Any, List, Optional, Union
import ast

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

from base_hf_dataset import BaseHFDataset, QAData

class HiddenFlawsGPT4VDataset(BaseHFDataset):
    """A class to handle loading and interacting with the Hidden Flaws GPT-4V dataset."""
    
    DATASET_NAME = "ncbi/Hidden-Flaws-GPT-4V"
    
    def __init__(self):
        """Initialize the dataset handler."""
        super().__init__(self.DATASET_NAME)
        
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
    dataset = HiddenFlawsGPT4VDataset()
    
    from dataset_exporter import DatasetExporter
    DatasetExporter.save(dataset, format='lmdb')
