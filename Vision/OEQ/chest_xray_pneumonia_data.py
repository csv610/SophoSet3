from typing import Dict, Any, List

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)



from base_hf_dataset import BaseHFDataset, QAData

class ChestXRayPneumoniaDataset(BaseHFDataset):
    """A class to handle loading and interacting with the Chest X-ray Pneumonia dataset."""
    
    DATASET_NAME = "hf-vision/chest-xray-pneumonia"
    
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
        question = "Is there evidence of pneumonia in this chest X-ray?"
        answer = "Normal" if row.get('label') == 0 else "Pneumonia"
        
        return QAData(
            key=self.get_key(index),
            question=question,
            answer=answer,
            image_path=row.get('image', '')
        )

if __name__ == "__main__":
    # Create the dataset
    dset = ChestXRayPneumoniaDataset()
    
    from dataset_exporter import DatasetExporter
    DatasetExporter.save(dset, format='lmdb')
