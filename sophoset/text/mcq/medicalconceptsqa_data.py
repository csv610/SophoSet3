from typing import Dict, Any, List, Optional

from sophoset.core.base_hf_dataset import BaseHFDataset, QAData

class MedicalConceptsQADataset(BaseHFDataset):
    """A class to handle loading and managing the MedicalConceptsQA dataset."""
    
    DATASET_NAME = "ofir408/MedConceptsQA"
    
    def __init__(self):
        """Initialize the MedicalConceptsQA dataset handler."""
        super().__init__(self.DATASET_NAME)
        
    def extract_row_data(self, row: Dict[str, Any], index: int) -> QAData:
        """Extract and format data from a MedicalConceptsQA dataset row.
        
        Args:
            row: The dataset row to extract data from
            index: The index of the row in the dataset
            
        Returns:
            QAData object containing the formatted row data
        """
        # Extract question and clean it
        question = row.get('question', '')
        
        # Extract options
        options = []
        for i in range(1, 6):  # Assuming there are up to 5 options (A-E)
            option = row.get(f'option_{i}', '')
            if option:
                options.append(option)
        
        # Extract answer
        answer = row.get('answer_id', '')
        
        return QAData(
            key=self.get_key(index),
            question=question,
            options=options,
            answer=answer
        )

if __name__ == "__main__":
    dset = MedicalConceptsQADataset()
    
    from sophoset.utils.dataset_exporter import DatasetExporter
    DatasetExporter.save(dset, format='lmdb')
