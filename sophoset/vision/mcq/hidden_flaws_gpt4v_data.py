from typing import Dict, Any, List, Optional, Union
import ast
import os
from pathlib import Path

from sophoset.core.base_hf_dataset import BaseHFDataset, QAData
from sophoset.utils.dataset_exporter import DatasetExporter
from sophoset.utils.dataset_explorer import DatasetExplorer

class HiddenFlawsGPT4VDataset(BaseHFDataset):
    """A class to handle loading and interacting with the Hidden Flaws GPT-4V dataset."""

    DATASET_NAME = "ncbi/Hidden-Flaws-GPT-4V"

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
        question = row.get('question', '')
        options_list = row.get('options', [])
        answer = row.get('answer', '')

        # Extract raw images - store as-is
        images = row.get('image', [])
        if not isinstance(images, list):
            images = [images] if images else []

        # Format options to dict with letter keys
        formatted_options = self.get_formatted_options(options_list)

        return QAData(
            key=self.get_key(index),
            question=question,
            options=formatted_options,
            answer=answer,
            images=images
        )


if __name__ == "__main__":
    dataset = HiddenFlawsGPT4VDataset()
    explorer = DatasetExplorer(dataset)
    for qa_data in explorer.next_question():
        explorer.print_question(qa_data)
