from typing import Dict, Any, Optional

from sophoset.core.base_hf_dataset import BaseHFDataset, QAData

class SciQDataset(BaseHFDataset):
    """A class to handle loading and managing the SciQ dataset."""
    
    DATASET_NAME = "allenai/sciq"
    
    def __init__(self):
        """Initialize the SciQ dataset handler.
        """
        super().__init__(self.DATASET_NAME)
    
    def extract_row_data(self, row: Dict[str, Any], index: int) -> QAData:
        """Extract and format data from a SciQ dataset row.
        
        Args:
            row: The dataset row to extract data from
            index: The index of the row in the dataset
            
        Returns:
            QAData object containing the formatted row data
        """
        # Extract question and options
        question = row.get('question', '')
        answer = row.get('correct_answer', '')
        
        # Get distractors and combine with correct answer
        options = [
            row.get('distractor1', ''),
            row.get('distractor2', ''),
            row.get('distractor3', ''),
            answer
        ]
        
        return QAData(
            key=self.get_key(index),
            question=question,
            options = options,
            answer="D"
        )

if __name__ == "__main__":
    dset = SciQDataset()
    
    from sophoset.utils.dataset_exporter import DatasetExporter
    DatasetExporter.save(dset, format='lmdb')
