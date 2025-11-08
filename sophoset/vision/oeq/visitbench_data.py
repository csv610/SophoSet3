from typing import Dict, Any, List, Optional

from sophoset.core.base_hf_dataset import BaseHFDataset, QAData

class VisitBenchDataset(BaseHFDataset):
    """A class to handle loading and interacting with the VisIT-Bench dataset."""
    
    DATASET_NAME = "mlfoundations/VisIT-Bench"
    
    def __init__(self):
        """Initialize the dataset handler."""
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
        # Extract basic information
        instruction = row.get('instruction', '')
        
        # Process image
        images = []
        image = row.get('image')
        if image is not None:
            try:
                img_data = self.get_image_data(image, format="PNG")
                images.append(img_data)
            except Exception as e:
                print(f"Error processing image: {str(e)}")
        
        # Format the question with any additional context
        question = instruction
        
        # Get reference answers if available
        reference_answers = row.get('reference_answers', [])
        answer = reference_answers[0] if reference_answers else ""
        
        # Create QAData object
        return QAData(
            key=self.get_key(index),
            question=question,
            images=images,
            answer=answer
        )


if __name__ == "__main__":
    # Create the dataset
    dset = VisitBenchDataset()
    
    from sophoset.utils.dataset_exporter import DatasetExporter
    DatasetExporter.save(dset, format='lmdb')
