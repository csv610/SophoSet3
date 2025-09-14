from typing import Dict, Any

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

from base_hf_dataset import BaseHFDataset, QAData

class GsmPlusDataset(BaseHFDataset):
    """A class to handle loading and managing the GSM-Plus dataset."""
    
    DATASET_NAME = "qintongli/GSM-Plus"
    
    def __init__(self):
        """Initialize the base class."""
        super().__init__(self.DATASET_NAME)
        
    def extract_row_data(self, row: Dict[str, Any], index: int) -> QAData:
        """Extract and format data from the dataset row.
        
        Args:
            row: The dataset row to extract data from
            index: The index of the row in the dataset
            
        Returns:
            QAData object containing the formatted row data
        """
        # Extract question and metadata
        question = row.get('question', '')
        answer = row.get('answer', '')
        explanation = row.get('solution', '')
        
        return QAData(
            key=self.get_key(index),
            question=question,
            answer=answer,
            explanation=explanation
        )

if __name__ == "__main__":
    dset = GsmPlusDataset()
    
    from dataset_exporter import DatasetExporter
    DatasetExporter.save(dset, format='lmdb')
