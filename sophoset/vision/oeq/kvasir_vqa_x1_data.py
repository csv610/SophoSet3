from typing import Dict, Any, List, Optional
import os
from pathlib import Path

from sophoset.core.base_hf_dataset import BaseHFDataset, QAData
from sophoset.utils.dataset_exporter import DatasetExporter
from sophoset.utils.dataset_explorer import DatasetExplorer

class KvasirVQAX1Dataset(BaseHFDataset):
    """A class to handle loading and interacting with the Kvasir VQA X1 dataset."""

    DATASET_NAME = "SimulaMet/Kvasir-VQA-x1"

    def __init__(self):
        """Initialize the Kvasir VQA X1 dataset handler.
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
        question = row.get('question', 'What is shown in this medical image?')
        answer = row.get('answer', '')

        # Extract raw images - store as-is
        images = row.get('image', [])
        if not isinstance(images, list):
            images = [images] if images else []

        return QAData(
            key=self.get_key(index),
            question=question,
            answer=answer,
            images=images
        )

if __name__ == "__main__":
    # Create the dataset
    dset = KvasirVQAX1Dataset()
    explorer = DatasetExplorer(dset)
    for qa_data in explorer.next_question():
        explorer.print_question(qa_data)
