from typing import Dict, Any, List, Optional
import os
from pathlib import Path

from sophoset.core.base_hf_dataset import BaseHFDataset, QAData
from sophoset.utils.dataset_exporter import DatasetExporter
from sophoset.utils.dataset_explorer import DatasetExplorer

class WorldMedQADataset(BaseHFDataset):
    """A class to handle loading and interacting with the WorldMedQA dataset."""

    DATASET_NAME = "WorldMedQA/V"

    def __init__(self):
        """Initialize the WorldMedQA dataset handler.
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
        options_list = []
        options_list.append(row.get("A", ""))
        options_list.append(row.get("B", ""))
        options_list.append(row.get("C", ""))
        options_list.append(row.get("D", ""))

        # Extract raw images - store as-is
        image = row.get('image', None)
        images = [image] if image else []

        answer = row.get("correct_option", "")

        # Format options to dict with letter keys
        formatted_options = self.get_formatted_options(options_list)

        return QAData(
            key=self.get_key(index),
            question=question,
            options=formatted_options,
            images=images,
            answer=answer
        )

if __name__ == "__main__":
    # Create the dataset
    dset = WorldMedQADataset()
    explorer = DatasetExplorer(dset)
    for qa_data in explorer.next_question():
        explorer.print_question(qa_data)
