from typing import Dict, Any, List, Optional

from sophoset.core.base_hf_dataset import BaseHFDataset, QAData

class ROCODataset(BaseHFDataset):
    """A class to handle loading and interacting with the ROCO-Radiology dataset."""
    
    DATASET_NAME = "mdwiratathya/ROCO-radiology"
    
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
        # Extract basic information
        answer = row.get('caption', '')
        
        # Create a detailed question prompt based on the caption
        question = (
            "Provide a detailed radiology report describing all relevant findings, "
            "including anatomical structures, potential abnormalities, and clinical significance. "
            "Be specific about locations, sizes, and any notable features."
        )
        
        # Process image
        images = []
        image = row.get('image')
        if image is not None:
            try:
                img_data = self.get_image_data(image, format="PNG")
                images.append(img_data)
            except Exception as e:
                print(f"Error processing image: {str(e)}")
        
        # Create QAData object
        return QAData(
            key=self.get_key(index),
            question=question,
            images=images,
            answer=answer 
        )


if __name__ == "__main__":
    # Create the dataset
    dset = ROCODataset()
    
    from sophoset.utils.dataset_exporter import DatasetExporter
    DatasetExporter.save(dset, format='lmdb')
