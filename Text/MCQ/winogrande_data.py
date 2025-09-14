from typing import Dict, Any, Optional

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

from base_hf_dataset import BaseHFDataset, QAData

class WinoGrandeDataset(BaseHFDataset):
    """A class to handle loading and managing the SciQ dataset."""
    
    DATASET_NAME = "allenai/winogrande"
    
    def __init__(self):
        """Initialize the SciQ dataset handler.
        """
        super().__init__(self.DATASET_NAME)
    
    def extract_row_data(self, row: Dict[str, Any], index: int) -> QAData:
        """Extract and format data from a SciQ dataset row.
        
        Args:
            row: The dataset row to extract data from
            index: The index of the row in the dataset
            
        Returns:
            QAData object containing the formatted row data
        """
        # Extract question and options
        question = row.get('sentence', '')
        
        # Get distractors and combine with correct answer
        options = [
            row.get('option1', ''),
            row.get('option2', '')
        ]
        answer = row.get('answer', '')

        try:
           answer = chr(ord('A') + int(answer)-1)
        except ValueError:
           answer = "-"
        
        return QAData(
            key=self.get_key(index),
            question=question,
            options = options,
            answer=answer
        )

if __name__ == "__main__":
    dset = WinoGrandeDataset()
    
    from dataset_exporter import DatasetExporter
    DatasetExporter.save(dset, format='lmdb')
