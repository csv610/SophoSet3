from typing import Dict, Any, List, Optional
import os
from pathlib import Path

from sophoset.core.base_hf_dataset import BaseHFDataset, QAData
from sophoset.utils.dataset_exporter import DatasetExporter
from sophoset.utils.dataset_explorer import DatasetExplorer

class VLMsAreBlindDataset(BaseHFDataset):
    """A class to handle loading and interacting with the VLMsAreBlind dataset."""

    DATASET_NAME = "XAI/vlmsareblind"

    def __init__(self):
        """Initialize the dataset handler."""
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
        # Extract basic information
        question = row.get('question', '')
        answer = row.get('answer', '')

        # Extract raw images - store as-is
        images = []
        image = row.get('image')
        if image is not None:
            images.append(image)

        # Create QAData object
        return QAData(
            key=self.get_key(index),
            question=question,
            images=images,
            answer=answer
        )

if __name__ == "__main__":
    # Create the dataset
    dset = VLMsAreBlindDataset()
    explorer = DatasetExplorer(dset)
    for qa_data in explorer.next_question():
        explorer.print_question(qa_data)
