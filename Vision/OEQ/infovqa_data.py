from typing import Dict, Any, List, Optional, Literal

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)


from base_hf_dataset import BaseHFDataset, QAData


class InfoVQADataset(BaseHFDataset):
    """A class to handle loading and interacting with the InfoVQA dataset."""
    
    DATASET_NAME = "LIME-DATA/infovqa"
    
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
        
        # Get the question and convert to lowercase for category determination
        question = row.get('question', '')
                
        # Get image data (handles both URLs and PIL Images)
        images = []
        if 'image' in row and row['image'] is not None:
            img_data = self.get_image_data(row['image'], format="PNG")
            images.append(img_data)

        # Extract answer if available (InfoVQA might have multiple answers)
        answers = row.get('answers', [])
        if not isinstance(answers, list):
            answers = []
        
        # Get the first answer if available, otherwise use an empty string
        answer = answers[0] if answers else ''
        
        return QAData(
            key=self.get_key(index),
            question=question,
            answer=answer,
            image_path=row.get('image', '')
        )


if __name__ == "__main__":
    # Create the dataset
    dset = InfoVQADataset()
    
    DatasetExporter.save(dset, format='json')
    DatasetExporter.save(dset, format='lmdb')
