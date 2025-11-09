from typing import Dict, Any, List, Optional, Literal
import os
from pathlib import Path

from sophoset.core.base_hf_dataset import BaseHFDataset, QAData
from sophoset.utils.dataset_exporter import DatasetExporter
from sophoset.utils.dataset_explorer import DatasetExplorer

class OCRBenchV2Dataset(BaseHFDataset):
    """A class to handle loading and interacting with the OCRBenchV2 dataset."""

    DATASET_NAME = "lmms-lab/OCRBench-v2"

    def __init__(self):
        """
        Initialize the dataset handler.
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
        question = "What text is visible in this image?"
        answer = row.get('text', '')

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
    dset = OCRBenchV2Dataset()
    explorer = DatasetExplorer(dset)
    for qa_data in explorer.next_question():
        explorer.print_question(qa_data)
