from typing import Dict, Any, List, Optional

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

from base_hf_dataset import BaseHFDataset, QAData

class GpqaDataset(BaseHFDataset):
    """A class to handle loading and managing the GPQA dataset."""
    
    DATASET_NAME = "Idavidrein/gpqa"
    
    def __init__(self):
        """Initialize the GPQA dataset handler.
        """
        super().__init__(self.DATASET_NAME)
    
    def extract_row_data(self, row: Dict[str, Any], index: int) -> QAData:
        """Extract and format data from a GPQA dataset row.
        
        Args:
            row: The dataset row to extract data from
            index: The index of the row in the dataset
            
        Returns:
            QAData object containing the formatted row data
        """
        # Extract question text
        question = row.get('Pre-Revision Question', '')
        answer   = row.get('Pre-Revision Correct Answer', '')
        explanation   = row.get('Pre-Revision Explanation', '')
        
        return QAData(
            key=self.get_key(index),
            question=question,
            answer=answer,
            explanation = explanation
        )

if __name__ == "__main__":
    dset = GpqaDataset()
    
    from dataset_exporter import DatasetExporter
    DatasetExporter.save(dset, format='lmdb')
