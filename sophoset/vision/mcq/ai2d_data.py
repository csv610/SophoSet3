from typing import Dict, Any, List, Optional
from pathlib import Path

from sophoset.core.base_hf_dataset import BaseHFDataset, QAData
from sophoset.utils.dataset_exporter import DatasetExporter
from sophoset.utils.dataset_explorer import DatasetExplorer

class AI2DDataset(BaseHFDataset):
    """A class to handle loading and interacting with the AI2D dataset."""

    DATASET_NAME = "lmms-lab/ai2d"

    def __init__(self):
        """Initialize the AI2D dataset handler.
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
        options_list = row.get('options', [])

        # Extract raw images - store as-is
        images = row.get('image', [])
        if not isinstance(images, list):
            images = [images] if images else []

        # Format options to dict with letter keys
        formatted_options = self.get_formatted_options(options_list)

        return QAData(
            key=self.get_key(index),
            question=question,
            images=images,
            options=formatted_options
        )

if __name__ == "__main__":
    dset = AI2DDataset()
    explorer = DatasetExplorer(dset)
    for qa_data in explorer.next_question():
        explorer.print_question(qa_data)

    
