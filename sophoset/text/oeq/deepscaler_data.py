from typing import Dict, Any, List, Optional

from sophoset.core.base_hf_dataset import BaseHFDataset, QAData

class DeepScaleRDataset(BaseHFDataset):
    """A class to handle loading and managing the DeepScaleR dataset."""
    
    DATASET_NAME = "agentica-org/DeepScaleR-Preview-Dataset"
    
    def __init__(self):
        """Initialize the DeepScaleR dataset handler."""
        super().__init__(self.DATASET_NAME)
    
    def extract_row_data(self, row: Dict[str, Any], index: int) -> QAData:
        """Extract and format data from a DeepScaleR dataset row.
        
        Args:
            row: The dataset row to extract data from
            index: The index of the row in the dataset
            
        Returns:
            QAData object containing the formatted row data
        """
        question = row.get('problem', '')
        answer = row.get('answer', '')
        explanation = row.get('solution', '')
        
        return QAData(
            key=self.get_key(index),
            question=question,
            answer=answer,
            explanation=explanation
        )

if __name__ == "__main__":
    dset = DeepScaleRDataset()
    
    from sophoset.utils.dataset_exporter import DatasetExporter
    DatasetExporter.save(dset, format='lmdb')
