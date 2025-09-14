from typing import Dict, Any, List, ClassVar

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

from base_hf_dataset import BaseHFDataset, QAData

class BlinkDataset(BaseHFDataset):
    """A class to handle loading and interacting with the BLINK dataset."""
    
    DATASET_NAME = "BLINK-Benchmark/BLINK"
    
    def __init__(self):
        """
        Initialize the dataset handler.

        """
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
        # Extract question and options
        question = row.get('question', '')
        options = row.get('options', [])
        
        # Extract images (context and candidate images)
        images = []
        """
        if 'context' in row and 'image' in row['context']:
            images.append(row['context']['image'])
        if 'candidate' in row and 'image' in row['candidate']:
            images.append(row['candidate']['image'])
        """
            
        # Get the correct answer
        answer = row.get('answer', "")
        
        return QAData(
            key=self.get_key(index),
            question=question,
            options=options,
            images=images,
            answer=answer
        )

if __name__ == "__main__":
    # Create the dataset
    dset = BlinkDataset()
    
    from dataset_exporter import DatasetExporter
    DatasetExporter.save(dset, format='lmdb')
