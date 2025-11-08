from typing import Dict, Any, List, Optional

from sophoset.core.base_hf_dataset import BaseHFDataset, QAData

class MMLUDataset(BaseHFDataset):
    DATASET_NAME = "cais/mmlu"
    
    def __init__(self):
        super().__init__(self.DATASET_NAME)
    
    def extract_row_data(self, row: Dict[str, Any], index: int) -> QAData:
        """Extract and format data from an MMLU-Pro dataset row.
        
        Args:
            row: The dataset row to extract data from
            index: The index of the row in the dataset
            
        Returns:
            QAData object containing the formatted row data
        """
        # Extract question, options, and answer
        question = row.get('question', '')
        options = row.get('choices', [])
        answer  = row.get('answer', '')

        try:
           answer = chr(ord('A') + int(answer))
        except ValueError:
           answer =  "NA"

        return QAData(
            key=self.get_key(index),
            question=question,
            options=options,
            answer=answer
        )

if __name__ == "__main__":
    dset = MMLUDataset()
    
    from sophoset.utils.dataset_exporter import DatasetExporter
    DatasetExporter.save(dset, format='lmdb')
