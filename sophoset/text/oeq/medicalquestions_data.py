from typing import Dict, Any, List, Optional

from sophoset.core.base_hf_dataset import BaseHFDataset, QAData

class MedicalQuestionsDataset(BaseHFDataset):
    """A class to handle loading and managing the MATH+ dataset."""
    
    DATASET_NAME = "fhirfly/medicalquestions"

    def __init__(self):
        """Initialize the MATH+ dataset handler."""
        super().__init__(self.DATASET_NAME)
    
    def extract_row_data(self, row: Dict[str, Any], index: int) -> QAData:
        """Extract and format data from a MATH+ dataset row.
        
        Args:
            row: The dataset row to extract data from
            index: The index of the row in the dataset
            
        Returns:
            QAData object containing the formatted row data
        """
        # Extract question and solution
        question = row.get('text', '')
        
        return QAData(
            key=self.get_key(index),
            question=question
        )

if __name__ == "__main__":
    dset = MedicalQuestionsDataset()
    
    from sophoset.utils.dataset_exporter import DatasetExporter
    DatasetExporter.save(dset, format='lmdb')
