from typing import Dict, Any, List, Optional

from sophoset.core.base_hf_dataset import BaseHFDataset, QAData

class MedicalMeadowFlashCardsDataset(BaseHFDataset):
    """A class to handle loading and managing the Medical Meadow Flash Cards dataset."""
    
    DATASET_NAME = "medalpaca/medical_meadow_medical_flashcards"

    def __init__(self):
        """Initialize the base class ."""
        super().__init__(self.DATASET_NAME)
    
    def extract_row_data(self, row: Dict[str, Any], index: int) -> QAData:
        """Extract and format data from the dataset row.
        
        Args:
            row: The dataset row to extract data from
            index: The index of the row in the dataset
            
        Returns:
            QAData object containing the formatted row data
        """
        # Extract question and solution
        question = row.get('input', '')
        answer   = row.get('output', '')
        
        return QAData(
            key=self.get_key(index),
            question=question,
            answer=answer,
            answer_choices=None
        )

if __name__ == "__main__":
    dset = MedicalMeadowFlashCardsDataset()
    
    from sophoset.utils.dataset_exporter import DatasetExporter
    DatasetExporter.save(dset, format='lmdb')
