from typing import Dict, Any, List, Optional

from sophoset.core.base_hf_dataset import BaseHFDataset, QAData

class PubMedQADataset(BaseHFDataset):
    """A class to handle loading and managing the PubMedQA dataset."""
    
    DATASET_NAME = "qiaojin/PubMedQA"
    
    def __init__(self):
        """Initialize the PubMedQA dataset handler.
        """
        super().__init__(self.DATASET_NAME)
    
    def extract_row_data(self, row: Dict[str, Any], index: int) -> QAData:
        """Extract and format data from a PubMedQA dataset row.
        
        Args:
            row: The dataset row to extract data from
            index: The index of the row in the dataset
            
        Returns:
            QAData object containing the formatted row data
        """
        # Extract question, context, and answer
        question = row.get('question', '')
        answer  = row.get('final_decision', '')
        explanation  = row.get('long_answer', '')
        
        return QAData(
            key=self.get_key(index),
            question=question,
            answer=answer,
            explanation= explanation
        )

if __name__ == "__main__":
    dset = PubMedQADataset()
    
    from sophoset.utils.dataset_exporter import DatasetExporter
    DatasetExporter.save(dset, format='lmdb')
