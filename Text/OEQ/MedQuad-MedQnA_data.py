from typing import Dict, Any, List, Optional

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from base_hf_dataset import BaseHFDataset, QAData

class MedQuadMedQnADataset(BaseHFDataset):
    """A class to handle loading and managing the MedQuad-MedicalQnA dataset."""
    
    DATASET_NAME = "keivalya/MedQuad-MedicalQnADataset"
    
    def __init__(self):
        """Initialize the MedQuad dataset handler."""
        super().__init__(self.DATASET_NAME)
    
    def extract_row_data(self, row: Dict[str, Any], index: int) -> QAData:
        """Extract and format data from a MedQuad dataset row.
        
        Args:
            row: The dataset row to extract data from
            index: The index of the row in the dataset
            
        Returns:
            QAData object containing the formatted row data
        """
        # Extract question and answer
        question = row.get('Question', '')
        answer = row.get('Answer', '')
        
        return QAData(
            key=self.get_key(index),
            question=question,
            answer=answer
        )

if __name__ == "__main__":
    dset = MedQuadMedQnADataset()
    
    from dataset_exporter import DatasetExporter
    DatasetExporter.save(dset, format='lmdb')
