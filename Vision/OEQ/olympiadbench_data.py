from typing import Dict, Any, List, Optional, Literal


import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)


from base_hf_dataset import BaseHFDataset, QAData

class OlympicArenaDataset(BaseHFDataset):
    """A class to handle loading and interacting with the OlympicArena dataset."""
    
    DATASET_NAME = "GAIR/OlympicArena"
    
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
        question = row.get('problem', '')
        answer = ''
        
        # Process image URLs
        image_path = ''
        figure_urls = row.get('figure_urls', [])
        if figure_urls:
            image_path = figure_urls[0]
        
        return QAData(
            key=self.get_key(index),
            question=question,
            answer=answer,
            image_path=image_path
        )


if __name__ == "__main__":
    # Create the dataset
    dset = OlympicArenaDataset()
    
    from dataset_exporter import DatasetExporter
    DatasetExporter.save(dset, format='lmdb')
