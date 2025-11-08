from typing import Dict, Any, List, Optional



from sophoset.core.base_hf_dataset import BaseHFDataset, QAData

class IAMSentencesDataset(BaseHFDataset):
    """A class to handle loading and interacting with the IAM Sentences dataset."""
    
    DATASET_NAME = "alpayariyak/IAM_Sentences"
    
    def __init__(self):
        """Initialize the dataset handler.
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
        question = "What text is written in this image?"
        answer = row.get('text', '')
        
        return QAData(
            key=self.get_key(index),
            question=question,
            answer=answer,
            image_path=row.get('image', '')
        )


if __name__ == "__main__":
    # Create the dataset
    dset = IAMSentencesDataset()
    
    from sophoset.utils.dataset_exporter import DatasetExporter
    DatasetExporter.save(dset, format='lmdb')
