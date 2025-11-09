from typing import Dict, Any, List, Optional
import os
from pathlib import Path

from sophoset.core.base_hf_dataset import BaseHFDataset, QAData
from sophoset.utils.dataset_exporter import DatasetExporter
from sophoset.utils.dataset_explorer import DatasetExplorer

class ROCODataset(BaseHFDataset):
    """A class to handle loading and interacting with the ROCO-Radiology dataset."""

    DATASET_NAME = "mdwiratathya/ROCO-radiology"

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
        # Extract basic information
        answer = row.get('caption', '')

        # Create a detailed question prompt based on the caption
        question = (
            "Provide a detailed radiology report describing all relevant findings, "
            "including anatomical structures, potential abnormalities, and clinical significance. "
            "Be specific about locations, sizes, and any notable features."
        )

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
    dset = ROCODataset()
    explorer = DatasetExplorer(dset)
    for qa_data in explorer.next_question():
        explorer.print_question(qa_data)
