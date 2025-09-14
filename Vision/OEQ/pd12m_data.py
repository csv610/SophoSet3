from typing import Dict, Any, List

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

from base_hf_dataset import BaseHFDataset, QAData
from dataset_exporter import DatasetExporter

class PD12MDataset(BaseHFDataset):
    """A class to handle loading and interacting with the CAMO dataset."""
    
    DATASET_NAME = "Spawning/PD12M"
    
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

        image = row.get('url', [])
        answer = row.get('caption', [])

        return QAData(
            key=self.get_key(index),
            question="Describe the image in detail",
            images=[image],
            answer=answer
        )

if __name__ == "__main__":
    # Create the dataset
    dset = PD12MDataset()
    
    DatasetExporter.save(dset, format='lmdb')
