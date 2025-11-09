from typing import Dict, Any, List, Optional, Literal
import os
from pathlib import Path

from sophoset.core.base_hf_dataset import BaseHFDataset, QAData
from sophoset.utils.dataset_exporter import DatasetExporter
from sophoset.utils.dataset_explorer import DatasetExplorer

class MathVisionDataset(BaseHFDataset):
    """A class to handle loading and interacting with the MathVision dataset."""

    DATASET_NAME = "MathLLMs/MathVision"

    def __init__(self, split: str = "test"):
        """
        Initialize the dataset handler.

        Args:
            split: The data split to use (default is 'test' as it's the main evaluation split)
        """
        super().__init__(self.DATASET_NAME)
        self.split = split

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
        choices_list = row.get('options', [])
        answer = row.get('answer', '')

        # Format the answer with the selected choice if available
        formatted_answer = answer
        if choices_list and answer.isdigit() and 0 <= int(answer) < len(choices_list):
            formatted_answer = f"{choices_list[int(answer)]} (Option {chr(65 + int(answer))})"

        # Extract raw images - store as-is
        images = []
        image = row.get('decoded_image')
        if image is not None:
            images.append(image)

        # Format options to dict with letter keys
        formatted_options = self.get_formatted_options(choices_list)

        return QAData(
            key=self.get_key(index),
            question=question,
            images=images,
            options=formatted_options,
            answer=formatted_answer,
        )


if __name__ == "__main__":
    # Create the dataset
    dataset = MathVisionDataset(split="test")
    explorer = DatasetExplorer(dataset)
    for qa_data in explorer.next_question():
        explorer.print_question(qa_data)
