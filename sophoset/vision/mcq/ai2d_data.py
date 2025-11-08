from typing import Dict, Any, List

from sophoset.core.base_hf_dataset import BaseHFDataset, QAData

class AI2DDataset(BaseHFDataset):
    """A class to handle loading and interacting with the AI2D dataset."""
    
    DATASET_NAME = "lmms-lab/ai2d"

    def __init__(self):
        """Initialize the GPQA dataset handler.
        """
        super().__init__(self.DATASET_NAME)

    def extract_row_data(self, row: Dict[str, Any], index: int) -> QAData:
        """
        Extract and format data from a dataset row.
        
        Args:
            row: The dataset row to extract data from
            index: The index of the row in the dataset
            
        Returns:
            QAData object containing the formatted row data
        """
        question = row.get('question', '')
        options = row.get('options', [])
#       images  = row.get('image', [])
        images  = []
        
        return QAData(
            key=self.get_key(index),
            question=question,
            options=options,
            images=images
        )

if __name__ == "__main__":
    # Create the dataset
    dset = AI2DDataset()
    
    # Save to JSON and LMDB
    from sophoset.utils.dataset_exporter import DatasetExporter
    DatasetExporter.save(dset)
