from typing import Dict, Any, List, Optional, Literal
import os
from pathlib import Path

from sophoset.core.base_hf_dataset import BaseHFDataset, QAData
from sophoset.utils.dataset_exporter import DatasetExporter
from sophoset.utils.dataset_explorer import DatasetExplorer


class InfoVQADataset(BaseHFDataset):
    """A class to handle loading and interacting with the InfoVQA dataset."""

    DATASET_NAME = "LIME-DATA/infovqa"

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

        # Get the question and convert to lowercase for category determination
        question = row.get('question', '')

        # Extract raw images - store as-is
        images = []
        if 'image' in row and row['image'] is not None:
            images.append(row['image'])

        # Extract answer if available (InfoVQA might have multiple answers)
        answers = row.get('answers', [])
        if not isinstance(answers, list):
            answers = []

        # Get the first answer if available, otherwise use an empty string
        answer = answers[0] if answers else ''

        return QAData(
            key=self.get_key(index),
            question=question,
            answer=answer,
            images=images
        )


if __name__ == "__main__":
    # Create the dataset
    dset = InfoVQADataset()
    explorer = DatasetExplorer(dset)
    for qa_data in explorer.next_question():
        explorer.print_question(qa_data)
