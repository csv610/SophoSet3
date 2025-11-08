from typing import Dict, Any, Optional

from sophoset.core.base_hf_dataset import BaseHFDataset, QAData

class SimpleQADataset(BaseHFDataset):
    """A class to handle loading and managing the MetaMathQA dataset."""
    
    DATASET_NAME = "basicv8vc/SimpleQA"
    
    def __init__(self):
        """Initialize the MetaMathQA dataset handler."""
        super().__init__(self.DATASET_NAME)
    
    def extract_row_data(self, row: Dict[str, Any], index: int) -> QAData:
        """Extract and format data from a MetaMathQA dataset row.
        
        Args:
            row: The dataset row to extract data from
            index: The index of the row in the dataset
            
        Returns:
            QAData object containing the formatted row data
        """
        # Extract question and solution
        question = row.get('problem', '')
        answer = row.get('answer', '')
        
        return QAData(
            key=self.get_key(index),
            question=question,
            answer=answer
        )

if __name__ == "__main__":
    dset = SimpleQADataset()
    
    from sophoset.utils.dataset_exporter import DatasetExporter
    DatasetExporter.save(dset, format='lmdb')
