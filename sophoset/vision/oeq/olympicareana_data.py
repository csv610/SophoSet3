from typing import Dict, Any, List, Optional
import os
from pathlib import Path

from sophoset.core.base_hf_dataset import BaseHFDataset, QAData
from sophoset.utils.dataset_exporter import DatasetExporter
from sophoset.utils.dataset_explorer import DatasetExplorer

class OlympicArenaDataset(BaseHFDataset):
    """A class to handle loading and interacting with the OlympicArena dataset."""

    DATASET_NAME = "GAIR/OlympicArena"

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
        question = row.get('problem', '')
        answer = ''  # This dataset doesn't seem to have answers in the row

        # Process image URLs - extract raw images
        images = []
        figure_urls = row.get('figure_urls', [])
        if figure_urls:
            # Store URLs as-is (they will be handled later)
            images = figure_urls if isinstance(figure_urls, list) else [figure_urls]

        return QAData(
            key=self.get_key(index),
            question=question,
            answer=answer,
            images=images
        )


if __name__ == "__main__":
    # Create the dataset
    dset = OlympicArenaDataset()
    explorer = DatasetExplorer(dset)
    for qa_data in explorer.next_question():
        explorer.print_question(qa_data)
