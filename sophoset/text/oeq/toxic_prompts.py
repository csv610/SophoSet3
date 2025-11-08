from typing import Dict, Any, List, Optional

import sys
import os
from sophoset.core.base_hf_dataset import BaseHFDataset, QAData

class ToxicPromptDataset(BaseHFDataset):
    """A class to handle loading and managing the AIME 2025 dataset."""
    
    DATASET_NAME = "bench-llms/or-bench-toxic-all"
    
    def __init__(self):
        """Initialize the AIME 2025 dataset handler.
        """
        super().__init__(self.DATASET_NAME)
        
    def extract_row_data(self, row: Dict[str, Any], index: int) -> QAData:
        """Extract and format data from a dataset row.
        
        Args:
            row: The dataset row to extract data from
            index: The index of the row in the dataset
            
        Returns:
            QAData object containing the formatted row data
        """
        # Extract question text
        question = row.get('prompt', '')
        # Extract answer if it exists
        answer = row.get('category', '')
        
        return QAData(
            key=self.get_key(index),
            question=question,
            answer=answer
        )

if __name__ == "__main__":
    dset = ToxicPromptDataset()
    
    from sophoset.utils.dataset_exporter import DatasetExporter
    DatasetExporter.save(dset, format='lmdb')
