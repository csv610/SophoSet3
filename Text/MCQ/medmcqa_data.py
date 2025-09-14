from typing import Dict, Any, List, Optional

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

from base_hf_dataset import BaseHFDataset, QAData

class MedMCQADataset(BaseHFDataset):
    """A class to handle loading and managing the MedMCQA dataset."""
    
    DATASET_NAME = "openlifescienceai/medmcqa"
    
    def __init__(self):
        """Initialize the MedMCQA dataset handler."""
        super().__init__(self.DATASET_NAME)
    
    def extract_row_data(self, row: Dict[str, Any], index: int) -> QAData:
        """Extract and format data from a MedMCQA dataset row.
        
        Args:
            row: The dataset row to extract data from
            index: The index of the row in the dataset
            
        Returns:
            QAData object containing the formatted row data
        """
        question = row.get('question', '')
        
        options = [row.get('opa', ''),row.get('opb', ''), row.get('opc', ''), row.get('opd', '')]
        
        # Get correct answer (0-3 maps to A-D)
        correct_idx = row.get('cop', -1)
        answer = chr(65 + correct_idx) if 0 <= correct_idx <= 3 else '?'
        
        return QAData(
            question=question,
            options = options,
            answer=answer,
            key=self.get_key(index)
        )

if __name__ == "__main__":
    dset = MedMCQADataset()
    
    from dataset_exporter import DatasetExporter
    DatasetExporter.save(dset, format='lmdb')
