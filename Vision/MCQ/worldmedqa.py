from typing import Dict, Any, List

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

from base_hf_dataset import BaseHFDataset, QAData

class WorldMedQADataset(BaseHFDataset):
    """A class to handle loading and interacting with the AI2D dataset."""
    
    DATASET_NAME = "WorldMedQA/V"

    def __init__(self):
        """Initialize the GPQA dataset handler.
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
        question = row.get('question', '')
        options = []
        options.append( row.get("A", ""))
        options.append( row.get("B", ""))
        options.append( row.get("C", ""))
        options.append( row.get("D", ""))
        image  = row.get('image', [])
        images  = [image]
        answer  = row.get("correct_option", "")
        
        return QAData(
            key=self.get_key(index),
            question=question,
            options=options,
#            images=images
            answer = answer
        )

if __name__ == "__main__":
    # Create the dataset
    dset = WorldMedQADataset()
    
    from dataset_exporter import DatasetExporter
    DatasetExporter.save(dset, format='lmdb')
