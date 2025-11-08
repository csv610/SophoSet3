from typing import Dict, Any, List, Optional


from sophoset.core.base_hf_dataset import BaseHFDataset, QAData

class MedicalQADataset(BaseHFDataset):
    """A class to handle loading and managing the Medical QA Datasets collection."""
    
    DATASET_NAME = "Malikeh1375/medical-question-answering-datasets"
    
    def __init__(self):
        """Initialize the Medical QA Datasets handler.
        """
        super().__init__(self.DATASET_NAME)
    
    def extract_row_data(self, row: Dict[str, Any], index: int) -> QAData:
        """Extract and format data from a Medical QA dataset row.
        
        Args:
            row: The dataset row to extract data from
            index: The index of the row in the dataset
            
        Returns:
            QAData object containing the formatted row data
        """
        # Extract question and answer
        question = row.get('input', '')
        answer = row.get('output', '')
        
        return QAData(
            key=self.get_key(index),
            question=question,
            answer=answer
        )

if __name__ == "__main__":
    dset = MedicalQADataset()
    
    from sophoset.utils.dataset_exporter import DatasetExporter
    DatasetExporter.save(dset, format='lmdb')
