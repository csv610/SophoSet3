from typing import Dict, Any, List, Optional, Literal

from sophoset.core.base_hf_dataset import BaseHFDataset, QAData

class MathVisionDataset(BaseHFDataset):
    """A class to handle loading and interacting with the MathVista dataset."""
    
    DATASET_NAME = "MathLLMs/MathVision"
    
    def __init__(self, split: str = "test"):
        """
        Initialize the dataset handler.
        
        Args:
            split: The data split to use (default is 'test' as it's the main evaluation split)
        """
        super().__init__(self.DATASET_NAME)
        self.split = split
    
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
        choices = row.get('options', [])
        answer = row.get('answer', '')
        
        # Format the answer with the selected choice if available
        formatted_answer = answer
        if choices and answer.isdigit() and 0 <= int(answer) < len(choices):
            formatted_answer = f"{choices[int(answer)]} (Option {chr(65 + int(answer))})"
        
        # Get image data (handles both URLs and PIL Images)
        images = []
        image = row.get('decoded_image')
        if image is not None:
            img_data = self.get_image_data(image, format="PNG")
            images.append(img_data)
        
        return QAData(
            key=self.get_key(index),
            question=question,
            images=images,
            options=choices,
            answer=formatted_answer,
        )


if __name__ == "__main__":
    # Create the dataset
    dataset = MathVisionDataset(split="test")
    
    from sophoset.utils.dataset_exporter import DatasetExporter
    DatasetExporter.save(dataset, format='lmdb')
