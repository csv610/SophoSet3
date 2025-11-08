from typing import Dict, Any, Optional

from sophoset.core.base_hf_dataset import BaseHFDataset, QAData
from sophoset.utils.dataset_exporter import DatasetExporter

class AimeDataset(BaseHFDataset):
    DATASET_NAME = "gneubig/aime-1983-2024"
    
    def __init__(self):
        super().__init__(self.DATASET_NAME)
    
    def extract_row_data(self, row: Dict[str, Any], index: int) -> QAData:
        """Extract and format data from an AIME dataset row.
        
        Args:
            row: The dataset row to extract data from
            index: The index of the row in the dataset
            
        Returns:
            QAData object containing the formatted row data
        """
        # Extract question text
        question = row.get('Question', '')
        
        # Extract answer if it exists
        answer = row.get('Answer', '')
        
        return QAData(
            key=self.get_key(index),
            question=question,
            answer=answer
        )

if __name__ == "__main__":
    dset = AimeDataset()
    
    DatasetExporter.save(dset, format='lmdb')
