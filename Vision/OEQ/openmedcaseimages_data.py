from typing import Dict, Any, List, Optional

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

from base_hf_dataset import BaseHFDataset, QAData

class OpenMedCaseImagesDataset(BaseHFDataset):
    """A class to handle loading and managing the TruthfulQA dataset."""
    
    DATASET_NAME = "openmed-community/multicare-case-images"

    
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
        question = row.get('caption', "")
        images = []
        image = row.get('image')
        if image is not None:
            try:
                img_data = self.get_image_data(image, format="PNG")
                images.append(img_data)
            except Exception as e:
                print(f"Error processing image: {str(e)}")

        return QAData(
            key=self.get_key(index),
            question=question,
            images  = images
        )

if __name__ == "__main__":
    # Create the dataset
    dset = OpenMedCaseImagesDataset()
    
    from dataset_exporter import DatasetExporter
    DatasetExporter.save(dset, format='lmdb')
