from typing import Dict, Any, List, Optional

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

from base_hf_dataset import BaseHFDataset, QAData

class OpenMedCasesDataset(BaseHFDataset):
    """A class to handle loading and managing the TruthfulQA dataset."""
    
    DATASET_NAME = "openmed-community/multicare-cases"

    
    def __init__(self):
        """Initialize the dataset handler.
        """
        super().__init__(self.DATASET_NAME)
    
    def extract_row_data(self, row: Dict[str, Any], index: int) -> QAData:
        """Extract and format data from a TruthfulQA dataset row.
        
        Args:
            row: The dataset row to extract data from
            index: The index of the row in the dataset
            
        Returns:
            QAData object containing the formatted row data
        """
        # Extract question and best answer
        cases = row.get('cases', [])
        question = ""
        if cases and isinstance(cases, list) and len(cases) > 0:
            # Get the first case's text if available
            first_case = cases[0]
            question = first_case.get("case_text", "")
        
        return QAData(
            key=self.get_key(index),
            question=question
        )

if __name__ == "__main__":
    dset = OpenMedCasesDataset()
    
    from dataset_exporter import DatasetExporter
    DatasetExporter.save(dset, format='lmdb')
